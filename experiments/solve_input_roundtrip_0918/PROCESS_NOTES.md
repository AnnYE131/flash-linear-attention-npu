# solve_input_roundtrip_0918 本地实现过程说明

执行者：OpenCode gateway/glm-5.3（A2 KKT后处理与Solve转换合并：独立本地实现）。
日期：2026-09-18。仅本地实现，无远端构建、无ATK、无递归委派。

## 起点

- 产品worktree：`A2SolveTri_Triton.ai-work/.runtime/solve-input-roundtrip-0918`
- 分支：`perf/gdn-a2-solve-input-roundtrip-0918`
- 基线：`2278df6a3a5b54240b89465996df8b8aa03eb73d`（起点核对HEAD一致、`git status`干净）
- 设计输入：`tasks/gdn-arch22-perf-opt/SOLVE_INPUT_FUSION_DRAFT_0918.md`（root编写，accepted-for-local-implementation）

## 实现前核对的事实（源码引证）

- `CopyOutTile`（chunk_scaled_dot_kkt.h，基线行316-339）：`OutputType != float` 时
  `Cast(typedOut, outTileLocal, CAST_RINT, valid*btAlign)` -> `PipeBarrier<PIPE_V>()` ->
  `CopyTypedOutTile`（`WaitVToMte3` -> `DataCopyPad(aGm)` -> `WaitMte3ToV`），typed写aWorkspace。
- `GdnFp32Solve::Run`（solve_tri_pipeline.h，基线行684-708）：入口无条件 `SyncAll<false>()`；
  `if constexpr (sizeof(In) != 4)` 内AIV执行 `full_convert<In>(raw, x, ...)`（CAST_NONE低精度->FP32）
  并随后再 `SyncAll<false>()`；其后各阶段同步均无条件。
- Phase6入口（chunk_gated_delta_rule_fwd_arch22.cpp，基线行273/313）：KKT以
  `ChunkScaledDotKkt<InputT, InputT>` 写aWorkspace；220分支以
  `Run<InputT, InputT>(aWorkspace, solveFp32InputOffset, ...)` 调用。
- host（chunk_gated_delta_rule_fwd_arch22_tiling.cpp行252/338-339）：`useFp32Solve`仅DAV_2201；
  solveFp32Input区预留 `B*Hv*T*BT*sizeof(float)`；aWorkspace预留 `B*Hv*T*BT*sizeof(uint16_t)`。
- `DTYPE_Q`仅FP16/BF16（aclnn入口与def检查），`btAlign = AlignUp(BT, 8)`，BT64/128下等于BT。
- 内部arch22 KKT头仅被 `chunk_cumsum_kkt_solve_tri.cpp` include；公开chunk_scaled_dot_kkt算子
  使用另一份独立头，不受本次改动影响。

## 实现内容（两个产品commit）

1. `249c1c09` chunk_scaled_dot_kkt.h：
   - 模板签名 `template <typename KType, typename OutputType = float, bool StoreFp32SolveInput = false>`，
     默认false保持既有全部实例行为；类级static_assert禁止 `StoreFp32SolveInput && OutputType==float`。
   - 新增公共 `InitSolveFp32Input(GM_ADDR)`：绑定独立 `GlobalTensor<float> solveInputGm`
     （长度 `B_*taskHeads_*T_*BT_`），仅flag为true时由Phase6调用。
   - `CopyOutTile` 新增首分支：Cast#1(RINT, FP32->OutputType) -> `PipeBarrier<PIPE_V>` ->
     Cast#2(NONE, OutputType->FP32，回写同一outTileLocal) -> `PipeBarrier<PIPE_V>` ->
     `CopyFp32OutTile`。
   - 新增 `CopyFp32OutTile`：镜像既有typed路径的 `WaitVToMte3 -> DataCopyPad(solveInputGm) ->
     WaitMte3ToV`，blockLen/srcStride/dstStride全部按 `sizeof(float)` 计算。
   - 无新增UB缓冲；`typedOutTileBuf_`/`outTileBuf_` 复用。
2. `c9476388` chunk_gated_delta_rule_fwd_arch22.cpp：
   - RunPhase6内 `#if __CCE_AICORE__ == 220` 定义 `constexpr bool kStoreFp32SolveInput`（220为true，
     其余架构false）。
   - KKT实例化为 `ChunkScaledDotKkt<InputT, InputT, kStoreFp32SolveInput>`；flag为true时调用
     `InitSolveFp32Input(userWorkspace + phase6->solveFp32InputOffset)`。
   - 220分支Solve调用改为 `GdnFp32Solve::Run<float, InputT>(solveFp32Input, solveFp32Input, ...)`
     （raw与x同为既有FP32输入地址；In=float编译期跳过full_convert与其后SyncAll，首SyncAll保留）。
   - 非220分支（`#else`）与所有host/tiling/公开算子零改动。

## 本地检查（无A2编译条件，未编译）

- `git diff` 逐行复核（见交付文档）；`git diff --check` 无空白错误。
- 括号平衡/无Tab/CRLF一致（两文件）机械检查通过。
- `git status` 仅含两目标文件；未触碰 `.runtime/kkt-share-0918`、tasks、他人实验目录。
- 详见 `evidence/solve-input-roundtrip-0918/GLM_LOCAL_DELIVERY.md`。

## 边界与后续

- 本实现为2278df6a上的独立转换消融，未带入KKT共享改动；与共享分支均修改
  chunk_scaled_dot_kkt.h 与 Phase6入口文件，后续合并需root处理文本/语义冲突。
- A2编译、ATK精度/性能、同输入ABBA证伪均待root接续。

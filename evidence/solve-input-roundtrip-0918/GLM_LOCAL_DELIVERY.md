# GLM_LOCAL_DELIVERY — A2 KKT后处理与Solve输入转换合并（独立本地实现）

- 执行者：OpenCode gateway/glm-5.3
- 日期：2026-09-18
- 产品worktree：`D:/workspace/大融合算子测试/gdntest/A2SolveTri_Triton.ai-work/.runtime/solve-input-roundtrip-0918`
- 分支：`perf/gdn-a2-solve-input-roundtrip-0918`

## 源码SHA

- 基线（未包含任何KKT共享改动）：`2278df6a3a5b54240b89465996df8b8aa03eb73d`
- commit 1（KKT头，FP32存储开关）：`249c1c09912e15e28156fb0bb6920444d2fe3f58`
- commit 2（Phase6入口启用+Run<float>）：`c9476388f0eab09507274ae61d21fa6c9dd373b3`（=当前HEAD）
- 交付时 `git status` 干净（本目录与experiments为旁车未跟踪新文件）。

## 变更范围（仅两个产品文件，两个独立小步commit）

1. `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_scaled_dot_kkt/op_kernel/chunk_scaled_dot_kkt.h`
   - 模板签名新增默认false开关：`template <typename KType, typename OutputType = float, bool StoreFp32SolveInput = false>`。
   - 类级 `static_assert(!StoreFp32SolveInput || !std::is_same_v<OutputType, float>)`：OutputType保持
     CAST_RINT舍入类型，FP32仅为GM存储类型，二者显式区分；float组合直接拒绝编译。
   - 新增 `InitSolveFp32Input(GM_ADDR)`（公共，仅flag开启时调用）与独立成员
     `GlobalTensor<float> solveInputGm`（长度 `B_*taskHeads_*T_*BT_`）。
   - `CopyOutTile` 首分支（仅 `StoreFp32SolveInput`）与新增私有 `CopyFp32OutTile`。
   - 既有typed路径、float路径、Init/InitFusedCumsum/InitCommon签名、UB缓冲分配全部未改。
2. `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp`
   - RunPhase6内按 `#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220` 定义
     `constexpr bool kStoreFp32SolveInput`（220=true，其他架构=false）。
   - KKT实例化 `ChunkScaledDotKkt<InputT, InputT, kStoreFp32SolveInput>`；flag开启时在Init后调用
     `InitSolveFp32Input(userWorkspace + phase6->solveFp32InputOffset)`。
   - 220分支Solve调用由 `Run<InputT, InputT>(aWorkspace, solveFp32InputOffset, ...)` 改为
     `Run<float, InputT>(solveFp32Input, solveFp32Input, ...)`；d16/d32/d64/y(A)/ws/cu/problem不变。
   - 非220 `#else` 分支、host tiling、tiling结构体、公开算子、arch35、README/API：零改动。

host侧事实（未改动，仅引证）：`useFp32Solve`仅DAV_2201置位并填写
`trailer.solveFp32InputOffset`（预留 `B*Hv*T*BT*sizeof(float)`）；`aWorkspaceBytes`
（`B*Hv*T*BT*sizeof(uint16_t)`）与x预留均未缩减；无新增序列化tiling字段。

## 两次Cast与存储/屏障证明（源码级，非硬件结论）

新路径每个有效KKT任务在 `CopyOutTile`（flag开启分支）执行：

1. `Cast(typedOut, outTileLocal, RoundMode::CAST_RINT, valid*btAlign_)`
   — typedOut位于独立 `typedOutTileBuf_`（与outTileBuf_不同TBuf、不同地址）。舍入语义与原typed路径
   完全相同，未省略。
2. `PipeBarrier<PIPE_V>()` — 第一次Cast（含对outTileLocal的全部读取）完成后才继续；与本仓既有
   依赖型V算子间协议一致（如 `ComputeEpilogueRow` 的 Duplicate->PipeBarrier->Mul 同缓冲覆盖模式）。
3. `Cast(outTileLocal, typedOut, RoundMode::CAST_NONE, valid*btAlign_)`
   — 第二次Cast在第一次读完后覆盖同一FP32槽；源为typedOut，BF16/FP16->FP32为精确拓宽。
   值恒等式（待硬件验证的目标合同）：新x有效行元素 ==
   FP32(CAST_RINT<InputT>(原epilogue FP32))，与旧full_convert输出逐位同构。
4. `PipeBarrier<PIPE_V>()` — 第二次Cast完成。
5. `CopyFp32OutTile`：`WaitVToMte3()`（V->MTE3交付）-> `DataCopyPad(solveInputGm[outBaseOffset],
   outTileLocal, outParams)` -> `WaitMte3ToV()`（MTE3->V归还，outTileLocal可被下一任务复用）。
   blockCount=valid（仅有效行），blockLen=BT_*4B，srcStride=(btAlign_-BT_)*4/32（BT64/128下=0），
   dstStride=(outRowStride-BT_)*4（=0，行连续）。GM元素偏移 `outBaseOffset=((b*taskHeads_+h)*T_+
   rowStart)*BT_` 与旧full_convert产生的x元素序完全一致（full_convert为逐元素稠密拷贝）。

跨核发布：KKT AIV的MTE3写入在 `WaitMte3ToV` 后于本核完成，`kktPipe.Reset()` 后进入
`GdnFp32Solve::Run`；Run入口第一个 `SyncAll<false>()`（源码注释即"KKT/staging 的跨组 GM 写入对
所有 Solve 消费者可见"）发布给全部AIC/AIV消费者。`In=float` 使 `if constexpr (sizeof(In) != 4)`
整体编译排除：full_convert与其后的第二个 `SyncAll<false>()` 消失；leaf/merge各阶段间及最终mixed
边界全部无条件保留。该"首SyncAll发布MTE3写入"的可见性合同与旧路径第二次SyncAll发布
full_convert写入属同一原语，未引入新的同步原语。

未写入区域审计：aWorkspace（低精度区）在新路径下无任何写入亦无读者；solveFp32Input仅写valid行，
varlen尾块与末chunk不足BT的行不写（blockCount=valid），不会覆盖下一序列；Solve侧
`locate`/`pos` 仅按cuSeqlens或T读取有效行，未读区域与旧路径同为未定义（旧路径为full_convert
搬运的workspace垃圾值，新路径为未写，二者均不被消费）。有效行对角/上三角零值仍来自
`ComputeEpilogueRow` 的Duplicate(0.0f)+前缀Mul，经RINT/NONE两次Cast后±0逐位保持（拓宽精确）。

UB容量：无新增InitBuffer；typedOutTileBuf_仅OutputType!=float时分配（原逻辑），第二次Cast复用
outTileBuf_，总UB占用不变。

## 路由核对（源码级）

- 四个入口key（host `TILING_KEY_V128=1/V256=2/PREPARED_BTH=3/PREPARED_BTH_V256=4`）全部进入
  `chunk_gated_delta_rule_fwd` -> `RunPhase6`；开关仅由 `__CCE_AICORE__==220` 决定，与key无关，
  即A2上四个key（V128/V256 × prepared/非prepared）均启用新路径，其他架构四个key行为与基线一致。
- FP16/BF16：OutputType=InputT两态均走同一两次Cast；static_assert仅排除float（DTYPE_Q上游仅
  允许FP16/BF16，合法配置不可能触发）。
- BT64/128：btAlign=AlignUp(BT,8)=BT；DataCopyPad行参按FP32计算对两BT均成立。
- 默认模板调用（standalone chunk_cumsum_kkt_solve_tri的 `ChunkScaledDotKkt<DTYPE_K, DTYPE_K>`、
  非220 Phase6的 `ChunkScaledDotKkt<InputT, InputT, false>`）编译期排除新分支，行为不变；
  公开chunk_scaled_dot_kkt算子使用独立头文件，不受影响。
- `InitSolveFp32Input`/`CopyFp32OutTile` 仅在flag开启时被实例化或调用。
- state_output host变体经主 `Tiling4ChunkGatedDeltaRuleFwdArch22` 复用并随后由主函数写Phase6
  trailer与最终tiling key，220下solveFp32InputOffset必有有效值（host既有逻辑，未改）。

## 真实本地检查结果（本机无A2编译条件，全程未编译）

- 起点核对：worktree HEAD=2278df6a、分支正确、状态干净。
- `git diff`（两commit前）逐行人工复核：无范围外改动。
- `git diff --check`：无空白/冲突标记问题。
- 机械检查：两文件花括号76/76、59/59平衡；无Tab；CRLF行尾一致。
- `git status`：仅两目标产品文件；kkt-share-0918、原A2SolveTri_Triton、tasks目录未触碰
  （旁车experiments/evidence为本任务新建目录）。
- 未做也无效：CPU公式复述自测不作为硬件精度或同步通过的宣称。

## 潜在边界与风险（供root评审）

1. 第二次Cast使KKT epilogue每任务多一次全tile V遍历；若MTE3 FP32写与该V遍历的代价超过
   省去的GM往返+full_convert，可能无收益——需同输入同版本ABBA证伪，未预估百分比。
2. `SyncAll<false>` 对"AIV MTE3写->AIC cube读"的可见性与旧路径第二次SyncAll同原语，但发布点
   提前到KKT直写；若A2上存在更弱的缓存可见性细分（未在仓内证据中出现），需ATK/msprof确认。
3. varlen下solveFp32Input未写行的内容与旧路径不同（旧为搬运的垃圾、新为未写）；已论证无读者，
   但内存覆盖审计用例如需逐位比较该区域应只按"有写入/有mask证明"判定。
4. aWorkspace继续预留但A2新路径下完全无写无读；后续若要缩减预留属host/布局变更，超出本任务。
5. 与KKT共享分支的合并：两边都修改 chunk_scaled_dot_kkt.h 与 chunk_gated_delta_rule_fwd_arch22.cpp
   （本侧新增第三模板参数与InitSolveFp32Input/CopyFp32OutTile），文本与语义冲突由root裁决，
   本任务不替root合并。

## 待A2构建/ATK事项（root接续）

- A2（DAV_2201）算子编译：四个tiling key × {BF16, FP16} × {BT64, BT128} × {V128, V256} ×
  {prepared, 非prepared} 的编译面确认（无本地编译条件，本交付未编译）。
- 精度门禁：按draft"精度证明目标"，对每个真实token行BT列验证
  新x == FP32(CAST_RINT<InputT>(原后处理FP32))，含BF16/FP16边界舍入数、±0、小值、上下界；
  不以NPU自身为golden。
- 性能：同输入同版本ABBA，绑定最终提交；不预设百分比。
- 正式MSS/门禁绑定最终提交。

# Finalize A2 开发期验证

## P117 当前候选（待设备验证）

基线P90 `0359a05`，独立 worktree `flash-linear-attention-npu-finalize-a2-varlen-scan`。
唯一改动是将现有 DG合并/FP32反向scan 扩展到host元数据定义的packed varlen。
源码与边界详见 design.md 的 P117 章节；无dtype降低、近似或模型shape分支。
原性能数据来自封存P116：C01 19258.13µs（限18684.85），C02 36496.57µs
（限27715.65）；C20 24775.04µs（限33248.83）。这些是共享卡参考，不是本轮新门禁。

| 模型case | 状态 | 触发证据 | 唯一变量 | 精度 | 性能前后 | 结论 |
|---|---|---|---|---|---|---|
| C02/C20及通用varlen | 待实测 | Add+旧scan，C02 scan约9.98ms | 融合scan支持序列元数据 | 待all6/ATK500/真FP64/DG | 待新同卡计时 | 不晋升 |
| C01及全部dense | 待回归 | 主体dense scan路径既有 | 只新增constexpr varlen分支 | 待原六/500 | 待全20 | 不晋升 |

阶段核验重点：所有有效token写一次；尾位置先置0再reverse scan；保持add存储舍入；
metadata cache按值/设备/stream分开，上传与读取同stream，启动前排空生产队列。
当前只完成实现分析与候选代码，编译UB和实际流水、精度/性能均未确认。

源码基线：`06ba242f253568031cb5ea18a59b102f9f9030ca`。当前目标是
在 Ascend 910B 上使同名融合核进入真实设备路径，先以
`atkFinalize双标杆/cases/bootstrap.json` 的三例验证公式。A5
路径保持原有 16 阶段代码与 TilingKey。

| 项目 | 当前实现 | 验证状态 |
| --- | --- | --- |
| A2 接口及支持组合 | `op_host/*def.cpp`、`_stable.py`、`_aclnn_ctypes.py` | 独立 wheel 编译与真实调用通过 |
| `(chunk,HK)` 所有权、GVA 归并 | `op_kernel/arch22/*reference.h` | ATK 五例通过 |
| 每 core 11 个 FP32 scratch 平面 | `op_host/op_tiling/*tiling.cpp` | 五例无越界/非有限；峰值与长序列待测 |
| dense/varlen、尾块、norm | `op_kernel/arch22/*reference.h` | ATK 五例通过 |
| A2 时间和 AIC/AIV 流水 | 当前仅标量 AIV 正确性版本 | 未达性能设计 |

正式构建、ATK 与性能结果记录在隔离旁车仓，不以旧链 bootstrap
结果代替真实 DUT 结果。每轮源码和安装产物需绑定 Git commit。

首轮 `11b6693` 在 A2 Bisheng 后端报
`not support bf16 type cast`，未安装、未运行。新版本改用 BF16
位展开和 round-to-nearest-even 打包，等待重新编译验证。

`40e062b` 已编译并安装到独立 site，真实 DUT 首次调用仍在
`_stable.py` 的旧 Ascend950-only 守卫处被拒绝，尚未进入设备核。
下一次构建须包含 Stable ABI 路由更新。

`ce7c9f9` 在 A2 NPU:2 的隔离 `site-atk-chain` 已完成五算子
wheel 构建、安装、真实 Finalize 运行。双标杆 ATK 的
`bootstrap.json` 三例和 `accuracy.json` 五例均返回
`result=passed`、`dut_mode=real`；五例各有
DUT/旧链/FP64 三角色且每例三角色输入哈希一致。独立 `run_npu.py`
逐例与 FP64 比较，所有输出有限，五例中最大的相对 RMS 为
`0.001859`（约 0.186%）。原始 manifest、role hash、summary
与逐例指标在同级旁车仓 `tasks/finalize-a2-661/evidence/runs/`。

这些结果仅证明正确性首版在所列小 shape 成立。当前标量 AIV
版本单次 Python launch+NPU sync 为 `dense_one≈186 ms`、
`dense_full_gva2_norm≈761 ms`、`dense_tail_gva2≈788 ms`、
`varlen_tail_gva3_norm≈1038 ms`、`dense_gva4_norm≈1370 ms`；
该指标不是设备 Task Duration，也不用于同基线 6192 µs
或 H20 2367 µs 作直接性能比。AIC/Cube、批量 GM 搬运、
双缓冲和长序列性能均未实现或验证。

P1 首轮 `024aa5b` 的正式 A2 构建失败，Catlass
`PackedTileCopyTla` 报架构分支模板形参错误，未安装或运行。
下一版在 A2 Cube 头文件引入 Catlass 前固定 `CATLASS_ARCH=2201`，
以实际编译结果检验该定位。

`a41a4c8` 已越过架构模板选择，但构建停在 TLA tile 坐标类型：
无前缀的 `MakeCoord` 被解析为 Catlass 坐标，而 `tla::GetTile`
要求 TLA tuple；同时 `GemmCoord` 命名空间写错。下一版对
`tla::MakeCoord/MakeShape/GetTile` 和 `Catlass::GemmCoord`
显式限定，再重建验证。

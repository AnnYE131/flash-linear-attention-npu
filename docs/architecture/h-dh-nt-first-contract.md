# h/dh NT-first 迁移契约与 P1 标杆

基线：PR #701 合并最新 main 后的 `b053cffb`（main `909db6ea`）。
状态：P1 准备完成；P2 前向接口已修改并进行离线检查，设备验证待执行；尚未全量迁移。
下表描述待实施的接口变化，不表示当前 wheel 已支持。

## 目标契约

| 场景 | h 与 dh 的相同形状 |
| --- | --- |
| dense，K-first | `[B,NT,HV,K,V]` |
| dense，V-first | `[B,NT,HV,V,K]` |
| packed varlen，K-first | `[total_NT,HV,K,V]` |
| packed varlen，V-first | `[total_NT,HV,V,K]` |

“rank-3 varlen”指 token 输入；状态张量自身为 rank-4。
NT 在 dense 中为每个 batch 的 `ceil(T/chunk_size)`；packed 中为
各序列 `ceil(length/chunk_size)` 之和，不能对 total_T 只做一次 ceil。
状态按 sequence-major chunk 顺序排列，value head 不映射成 query/key head。
`state_v_first` 只改变 K/V 顺序，不改变 NT/HV 顺序或数学语义。

packed 公开状态去掉物理 batch=1。内部 L0 若仍需 rank-5，允许无复制的
`unsqueeze/reshape` 视图；这不构成另一种公开布局。token 本身的布局契约不变。
旧 head-first 不根据 shape 自动猜测；HV=NT 时必须由调用者按版本传入正确数据。
h0/ht/dh0/dht 没有 chunk 轴，维持原语义；新增布局不能改变初态/末态的位置。

支持平台、dtype、K/V、chunk_size、gate 选项与可选输入仍以各算子的现有 API
为边界。原本不支持 state_v_first 的配置继续拒绝，不借本次换轴扩大支持域。
已有该开关的路径需要同步 h/dh 的实际末维存储，不能只改 descriptor。

## 受影响入口登记

本表中“迁移”包含对应 ACLNN、tiling、Stable ABI、ctypes 以及相同 kernel 的调用者。
各算子 API 文件中的“NT-first 迁移目标”指向本文；完成设备修改时更新其正式 shape 表。

| 算子 | 状态角色 | 当前基线与目标差异 |
| --- | --- | --- |
| chunk_fwd_h | 写 h | P2：packed 公开输出已去除首维 1 |
| chunk_gated_delta_rule_fwd_h | 写 h | head-first → NT-first，普通/preload 和平台副本一致 |
| chunk_fwd_o | 读 h | P2：公开 packed rank-4，内部 rank-5 视图 |
| chunk_gated_delta_rule_fwd | 内部写/读并可导出 h | P2/P4：A5 导出与旧融合副本均已同步 |
| chunk_kda_fwd | 内部写/读并可导出 h | P4：V2 与旧融合内部均 NT-first，删除导出换轴 |
| chunk_kda_fwd_finalize | 读 h | P2：公开 packed rank-4，内部 rank-5 视图 |
| chunk_gated_delta_rule_bwd_dhu | 写 dh | head-first → NT-first；支持域内 state_v_first 应同时作用于 dh |
| chunk_bwd_dqkwg | 读 h/dh | 二者同步迁移 |
| chunk_gated_delta_rule_bwd_finalize | 读 h/dh | 二者同步迁移 |
| chunk_gated_delta_rule_bwd | 内部写/读 h/dh | P3：分配统一，已删除 hHead 转换 |
| chunk_kda_bwd | 读 saved h，内部写/读 dh | h 已统一；内嵌 state_scan、WyFinalize、V2 dh 分配共同迁移 |
| chunk_kda_bwd_prepare | 读 h | 保留已有 NT-first，核对 packed 入口 |
| chunk_kda_bwd_finalize | 读 h/dh | 保留 h，迁移 dh 与 host shape |

## CPU 标杆准备

### P2 前向实现记录（2026-09-22）

- Stable ABI 与 ctypes 同步修复 GDN h 导出的 NT/HV 顺序，并按 cu_seqlens 分配 packed rank-4。
- 共享 FwdH 公开输出已为 packed rank-4；FwdO 和 KDA Finalize 的 ACLNN 校验该形状，
  连续化后以补 B=1 的 reshape 视图进入原 rank-5 L0/tiling。没有新增 NT/HV 转置。
- FwdH 的逻辑维度来自属性，输出 rank 不参与地址计算；原 NT-first kernel 保持。
  KDA V2 的 FwdH→Finalize、bwd saved/recompute 均走内部 L0 rank-5 视图，保持原路径。
- 共享 FwdH ATK 默认输出切换到 packed rank-4。Finalize generator/executor 同步；
  accuracy/perf/mss 仍为 624/10/12 例，只有其中 216/2/10 例的 h shape 去掉首维 1。
  seed、值域、阈值和其他字段逐项保持一致。
- 旧独立 FwdH 留待 P4；其调用 FwdO 的 example/benchmark/PTA 临时保留换轴，
  varlen 再 squeeze(0)。fast launch 的内部 rank-5 descriptor 不变。

离线通过：`tests/test_nt_first_cpu_references.py` 6 组、
`tests/test_nt_first_forward_contract.py` 4 组（46 个参数场景）；后者仅执行真实 Python
分配/CPU 标杆，在 ACLNN launch 边界停止，不验证 C++、tiling 或 NPU 结果。
Stable gates 23 通过、1 跳过；33 个 ABI adapter 零不匹配；coverage/fallback 检查通过。

设备待执行：原有 FwdH/FwdO/Finalize/GDN export 专项，以及新增
`tests/stable_abi/test_forward_h_chain_nt_first.py`（A5 四例直接传 h）。两种后端各运行一次，
并保留 KDA saved/recompute 回归。OPP/wheel 未构建；NPU 精度、内存与性能尚无新结论。
FwdO 的完整 metadata/NT 容量拒绝检查仍在 P5 统一收敛范围内。

P1 不改 device kernel，也不提前切换现有 ATK 默认布局。以下显式选项仅用于
验证目标契约；P2–P4 对应设备迁移时再切换 executor 默认调用与冻结用例。
布局变换保持原有 reference 的公式、计算顺序、dtype/cast 与阈值。

| 语义 | 原始依据 / CPU 函数 | P1 操作 |
| --- | --- | --- |
| 独立旧 FwdH 每块起始状态 | `tests/atk/chunk_gated_delta_rule_fwd_h/executor_chunk_gated_delta_rule_fwd_h.py::_forward_h_ref` | `nt_first=True` 直接按新轴写 h；原数学递推保持 |
| 共享 FwdH packed 输出 | `tests/atk/chunk_fwd_h/executor_chunk_fwd_h.py::_reference` | `packed_output=True` 返回 rank-4 packed h，v_new/ht 不变 |
| Dhu 的逐 chunk 状态梯度 | `torch_custom/fla_npu/test/test_bwd_dhu.py::chunk_gated_delta_rule_bwd_dhu_cpu` | `nt_first=True` 直接按新轴写 dh；packed squeeze、末维属性生效 |
| GDN Finalize 读取 h/dh | `tests/atk/chunk_gated_delta_rule_bwd_finalize/scripts/chunk_gated_delta_rule_bwd_finalize_cpu.py` | `nt_first=True` 修改 shape 校验、状态索引与临时状态轴长度 |
| dqkwg 的 h/dh 消费 | `tests/atk/chunk_bwd_dqkwg/executor_chunk_bwd_dqkwg.py::chunk_bwd_dqkwg_torch` | `nt_first=True` 直接交给原已 NT-first 的内部 CPU 公式；packed 恢复 B=1 视图 |
| KDA saved h / Finalize | `tests/atk/chunk_kda_fwd/executor_chunk_kda_fwd.py::_reference_impl`、`chunk_kda_fwd_finalize/executor_chunk_kda_fwd_finalize.py::run_cpu` | 原已按 NT-first 计算，保留；packed Finalize 的入口适配随 P2 同步 |

KDA/GDN 组合标杆继续复用其原算子标杆；对应生产/消费都迁移后，再删除 CPU
组合中的旧布局转换。KDA 融合 dh 不新增一套数学公式，用独立 Dhu 与完整反向
回归验证。仓内本次检索未发现 GPU dump loader；外部 dump 验证在设备阶段登记，
不能把本次 CPU 等价测试称为 GPU 双标杆。

## 离线检查及限制

```sh
python tests/test_nt_first_cpu_references.py
```

该测试从原文件加载纯函数/常量定义，绕开 ATK/PTA 的框架导入副作用；使用真正
CPU PyTorch 张量执行既有 reference，没有 mock 数学运算或加载 NPU 输出。
dense B>1、HV!=NT/HV==NT、尾块、packed 非等长和 K!=V 的小规模地址代数用例，
均比较布局归一化后的完整输出，要求 shape/dtype 一致、有限且逐元素完全相等。
小 K/V 用例仅验证 CPU 索引，不代表设备支持域扩展。部分检查用三个固定 seed；
共享 FwdH 与 Finalize 保持其固定 128 维配置。

额外的等轴负例确认：同 shape 的错误 head/chunk 排列会改变 Finalize 结果。
既有正式 ATK 的随机输入、值域和容差没有改动，未重新冻结用例。

P3 已补齐 Dhu reference 的 `dht` 种子，并以独立前向递推的 PyTorch autograd
核对 dense/packed 和两种末维布局。设备 Dhu 入口仍忽略 `dht`，这是原有功能缺口，
尚未修复；带非零 dht 的设备用例不能计为通过。CPU 布局等价不代替数值精度、
内存检测、CT/ATK 验收或任何平台的设备通过结论。

## GDN h 导出失败用例

```sh
python -m pytest -v tests/stable_abi/test_gdn_h_export_nt_first.py
FLA_NPU_STABLE_ABI=ctypes python -m pytest -v tests/stable_abi/test_gdn_h_export_nt_first.py
```

在 A5 上覆盖 dense HV=2/3、NT=3，多 batch；packed lengths=[1,64,65]，
total_NT=4；两种末维顺序与三个 seed。利用 beta=0、g=0 时状态保持初态的
数学恒等式直接检查 h 和 ht；随机非对称初态区分 batch/sequence/head/K/V。
该用例不 xfail，当前生产代码的 wrapper 漏改及 packed rank 问题应明确失败。
P1 只固化用例，设备执行按约定延后到实现完成。

## P1 离线执行记录（2026-09-22）

- 环境：Windows、Python 3.12.10、隔离环境中的 PyTorch 2.14.0+cpu。
- CPU 布局等价：6 组测试、29 组参数场景通过，另包含等轴错读负例。
- 7 个新增/修改 Python 文件语法检查通过，11 个迁移文档链接有效。
- Stable ABI 离线门禁：24 项，23 通过、1 跳过。
- 未运行新增 A5 h 导出用例，未运行 ATK/CT 或任何 NPU；这些结果不计入设备验收。

## P3 反向迁移实现记录（2026-09-22）

GDN Dhu、dqkwg、BwdFinalize、顶层 backward，以及 KDA 两套内嵌 state_scan、
WyFinalize、独立 BwdFinalize、V2 backward 同步采用 NT-first。公开 dense 状态为
`[B,NT,HV,K,V]`，packed 为 `[totalNT,HV,K,V]`；支持 state_v_first 的路径交换末两维。
dqkwg 的两套 tiling 均已同步；ACLNN/fast launch 将 packed 输入补为内部 rank-5
无复制视图。GDN 组合入口删除 hHead 换轴。KDA dhHeadMajor 字段保留结构位置但固定为零。

### Dhu V-first 搬运与资源

Vector 复用原 dh0 的 16 行转置 helper 写出每个 chunk 的 dh；普通和 arch35 副本一致。
两个既有 output UB buffer 各为 `max(vecRow,16)*max(K,V)*sizeof(DT)`，同时承载
cast 后源矩阵与转置目标；本次不新增 UB、workspace 或 event。FP32 递推状态保持 K-first。
不足 16 行时，源地址表的无效行指向有效第 0 行，GM 只写有效行；UB 每个转置行
占一个 32B block，因此 UB→GM 的 srcStride 为 0，GM dstStride 为 `(K-curRows)*sizeof(DT)`。

复用前等待两个 output buffer 的 MTE3→V；Cast 后 PIPE_V barrier；转置后 V→MTE3；
每次搬运后发布 MTE3→V，下一 tile 覆盖目标前等待。既有跨核完成通知继续保护 Cube 读回。
Cube 根据 stateVFirst 选择现有 PackedTileCopyTla 的 RowMajor/ColumnMajor 状态类型，
逻辑矩阵保持 K×V；L1/L0 类型从该组件派生。fast launch 未暴露 V-first，保持默认 false。
新增模板实例必须在 A2/A5 编译验收，不能以离线检查代替。

### 验证与剩余项

- `tests/test_nt_first_backward_contract.py`：16 组真实 ctypes 分配与 4 组独立 autograd 标杆核对。
  仅 stub ACLNN launch，不模拟张量计算。
- `tests/stable_abi/test_backward_nt_first.py`：新增 16 组设备用例，使用 w=0 的独立恒等式，
  非零 k 检验 Dhu 的 Cube 读回；覆盖多 batch、非等长 packed、NT/HV 相等与不等、K!=V、两种末维布局。
- GDN Dhu/dqkwg/Finalize/组合 ATK executor、Finalize 冻结用例、fast launch 标杆及 PTA 已同步。
  C++ dqkwg 二进制示例需使用新 NT-first h.bin/dh.bin，旧 head-first 文件须重新生成或换轴后保存。
- Dhu 非零 dht 的设备功能缺口单独保留，不用全零末态梯度用例掩盖；本次 CPU 标杆可用于后续修复。
- 旧独立 FwdH 在 P4 迁移。example 中从它进入新 dqkwg 的 h 暂时显式换轴并去掉 packed batch 维。

设备待执行：Dhu→dqkwg、Dhu→GDN/KDA BwdFinalize、KDA 内嵌 state_scan→WyFinalize，
saved/recompute、端到端梯度、ATK 双标杆及内存检查。Stable ABI 与 ctypes 各运行一次，
fast launch 与原 ACLNN 各自回归。未编译 CANN/OPP/wheel，未运行 NPU，P3 阶段出口尚未验收。

## P4/P5 实现与跨入口检查（2026-09-22）

旧独立 FwdH 的普通/preload、arch35 副本，以及融合 GDN 的普通/arch22/arch35 FwdH
现已同步首状态地址、当前状态读址和下一块步长；内嵌 FwdO 同时迁移。
KDA 七份 prepare/post_wu/finalize/FwdH 状态地址同步，L2 与 direct launch 分配同步。
KDA 导出删除 head/chunk 转置，V-first 末维转换保留。UB/L1/L0、workspace 容量、任务分配
和事件同步未改变。旧 FwdH 详设见其 `docs/design.md`。

Stable ABI 的 `chunk_first` 临时参数已删除，两种 FwdH 分配规则相同。
example、six-aclnn benchmark、FwdO generalization 测试直接传 h，移除 P2/P3 临时适配。
保留的测试转换仅用于旧格式 CPU/reference fixture，不能用于新算子输出。

| 入口 | 接口、分配、生产/消费核查 | 标杆和测试 |
| --- | --- | --- |
| 共享 FwdH | dense 5D / packed 4D；原生 NT-first；内部 B=1 视图 | P1/P2 CPU、前向链路 |
| 旧独立 FwdH | ACLNN 精确 NT/rank；Stable/ctypes/fast 分配一致；7 份 scheduler 同步 | ATK 默认 NT-first；PTA/fast golden 更新；两种 producer 链路 |
| FwdO | ACLNN 校验 rank/head/尾维；共享 tiling 校验 NT；fast packed 4D | 原精度用例；直接 ctypes→ACLNN 正例及 4 个错误状态 descriptor 用例 |
| GDN 融合 fwd | A5 组合与旧融合内部一致；旧路径不新增 h 导出能力 | GDN export、six-aclnn 和既有 ATK |
| KDA 融合/V2 fwd | 七份 HOffset；导出不换 NT/HV；direct 分配 NT-first | 原 KDA ATK、direct 对比与地址表达式检查 |
| KDA FwdFinalize | P2 packed 4D、内部 B=1 视图、metadata 精确校验 | P2 ATK 冻结用例及链路 |
| Dhu | P3 NT-first/V-first；沿用原 metadata 校验 | CPU/autograd（dht=None）、Dhu 设备专项、fast/PTA |
| dqkwg | P3 两套 tiling、packed 视图；沿用原 metadata 校验 | 原 ATK/fast/PTA；旧 fixture 转换有显式说明 |
| GDN BwdFinalize/顶层 bwd | P3 直连，保留必要的 packed 视图 | P3 ATK 与组合 CPU 标杆 |
| KDA BwdFinalize/V2/fused bwd | P3 已统一；沿用原 metadata 校验 | saved/recompute 与端到端反向待 P6 |
| KDA BwdPrepare | 原已 NT-first；现有 rank/容量/canonical metadata 校验完整，允许空序列条目 | 保留原接口和用例，设备链路待 P6 |

验证前按最小修改原则复查，撤回 P5 新增的公共 metadata helper 及其调用和专属单测。
各入口沿用原有 metadata 校验，不统一空序列支持域；保留 h/dh 的 rank、轴序和容量检查。
KDA Prepare 原有完整 metadata 校验保留。
内部 L0 接口仍允许既有 rank-5 视图，不对外提供另一种 packed 存储。
撤回 P3 新增的 CPU dht 功能，独立 autograd 检查改用 dht=None，不扩展本次功能范围。

离线已执行：CPU reference 6 组、forward contract 4 组（新增旧 FwdH 的 4 场景）、
backward contract 2 组、真实源码地址表达式 2 组。地址测试覆盖 7 份 GDN scheduler
和 6 份 KDA HOffset，比较独立张量索引，包括 NT=HV 和 packed 全局 chunk 起点。
Stable gates 23 通过、1 跳过；33 ABI adapter 无 mismatch。未改冻结用例的 seed/值域/阈值。
本轮旧 FwdH ATK 以 case_spec 生成输出，不编码输出 h shape，因此无需重生成 JSON。

未执行：本地没有 C++/CANN 编译器，OPP/wheel 编译和所有 NPU 专项仍待 P6。
直接 ACLNN 拒绝、tiling 实际分支、普通/preload、A2/A3/A5 精度/内存/性能均不能计为通过。
P3 记录的 dht 功能缺口保持独立；本轮不宣称其已修复。

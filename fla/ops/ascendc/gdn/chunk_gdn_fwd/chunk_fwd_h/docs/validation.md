# NT-first 布局开发验证

基线：upstream main 82523d12。仅修改 ChunkFwdH 历史 h 的 chunk/head 顺序，
以及 KDA V2 重计算分配并删除对应转置；数学公式、dtype、属性、末两维布局不变。
本次仅打通该 h 链路，dh/旧独立 FwdH 和其他消费者 kernel 不迁移。

CPU 标杆沿用 executor_chunk_fwd_h.py 的公式、值域和容差，只更改 h 分配与保存下标。
先比较同输入下旧标杆转置与新标杆，要求 h 精确相同、v_new/final_state 原样相同。

验证计划：

- 当前源码完整构建目标链路 wheel，安装至任务隔离目录，确认实际加载产物。
- FwdH：g/gk、dense/packed、H=C/H!=C、B>1、尾块、初态/终态、state_v_first 两值。
- KDA：保存与重计算直接接通，dense/packed、H=C/H!=C、较大形状；对比梯度与确定性。
- 三个兼容 L2：KDA 前向、GDN 前向/反向回归；旧独立 FwdH 的 wrapper shape 保持。
- profiling 检查反向重计算中的 h Transpose/Contiguous 消失。
- ATK 正式验收及受支持 A2/A5 路径，不能以单纯 smoke 宣称全部通过。

实施追溯：公共 FwdHHOffset 覆盖 arch22/arch35 四个 Cube/Vector 文件，矩阵内部读写、
同步和容量不变。Stable ABI 分配器新增私有 chunk_first 选择，只有 ChunkFwdH 启用，
不改变公开 schema。其他组合入口只在 L2 做兼容转换，不修改其 kernel。

环境：Ascend950、CANN 9.1.0、Python 3.10.20。使用已有的 ATK 26.8.8 源码，
wheel 安装到任务隔离目录；不覆盖原环境安装。生产代码及测试共 11 个文件的规范化
SHA256 与构建机源码一致。

2026-09-20 已完成的开发验证（不是全部受影响算子的正式验收结论）：

- CPU 新旧标杆 24 组逐元素相同：新 h 等于旧 h 的 chunk/head 转置，v_new/final_state 不变。
- `H=NT=3` 的错误 chunk/head 映射在两种 state layout 下均被专项容差拒绝，避免只验 shape。
- A5 FwdH 专项 24 组通过，覆盖 g/gk、dense/packed、H=C/H!=C、B>1、尾块和 state_v_first。
- 完整 wheel 的专项共 30 组，Stable ABI 与 ctypes 两套入口均 30/30 通过；包含 6 组 KDA
  保存/重算对比与重复执行，覆盖 dense/packed 和 `B=1,H=64,T=4096` 三阶段前向链路。
- 共享及旧独立 FwdH 的两入口对比通过；`save_new_value=false` 两入口均按既有契约拒绝。
- GDN 反向 12 个既有 profile 的独立 CPU 标杆回归通过，覆盖 GVA、layout、tail、state/dht、
  varlen、norm/beta 和 state_v_first；仅为本次兼容适配的开发回归。
- GDN 前向 4 组 FP64 标杆回归通过，覆盖 dense、varlen、initial/final state、B>1、GVA 和 H=NT。
- A5 FwdH ATK accuracy 200/200 通过，启用 GM 初始化与 NaN 检测，沿用原值域和 mixed_tolerance_bm。
- A5 FwdH ATK memcheck 32/32 通过。汇总脚本未找到报告，直接核验生成的 XLSX：
  statistic 中 32 个不同 case 的内存检测均为 true、运行结果均为 SUCCESS，summary 为 100% Pass。
- A5 FwdH ATK determinism 32/32 通过，每例 50 次；首次因 ATK 队列 JSONDecodeError 未完成汇总，
  原样重试后 XLSX 的 summary 为 Pass、32 个用例均 SUCCESS。44 组 performance 采集完成，
  无失败用例；没有同条件修改前基线，不给出性能提升结论。
- KDA 重算对 FP64 saved-cache 标杆的 4 组 dense/packed 对比通过；该补充检查不替代 ATK 混合容差验收。
- KDA V2 重算 `B=1,H=64,T=4096,K=V=128` 的 msprof 采集共 10 次调用、50 个任务。
  每次为 Recompute、FwdH、Prepare、Dhu、Finalize，未出现额外 Transpose/Contiguous 任务。
  本次只确认任务链路，不据此宣称性能收益。

复现专项：安装当前分支构建的 wheel 后运行
`python -m pytest -v tests/stable_abi/test_fwd_h_nt_first.py`。
设置 `FLA_NPU_STABLE_ABI=ctypes` 可复验 ctypes 入口；默认入口为 Stable ABI。
裁剪构建必须显式包含组合算子的子算子二进制（不仅是 host 依赖）：
`chunk_fwd_h,chunk_kda_bwd,chunk_kda_fwd,chunk_kda_fwd_prepare,chunk_kda_fwd_finalize,chunk_gated_delta_rule_fwd,chunk_gated_delta_rule_bwd,chunk_gated_delta_rule_fwd_h,chunk_gdn_bwd_intra,chunk_gated_delta_rule_bwd_finalize,chunk_gated_delta_rule_fwd_prepare,chunk_fwd_o`。

完整 wheel 的 `libcust_opapi.so`、59 个 kernel .o 和 Python wrapper 均通过构建/安装一致性校验。
wheel SHA256：`95322e867134a2fdf447def6b20e476f9a0fe2adb3fbfa1fd064e30808c0d6f3`。

仍待完成：A2/A3 和全部受影响算子的正式验收及完整合入 CI；
保留本开发记录，不标记阶段 5 完成。

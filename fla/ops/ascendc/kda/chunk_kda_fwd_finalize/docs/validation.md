# NT-first 第二批开发验证

基线 `7516a589`，范围为共享 FwdH → KDA V2 → 独立 Finalize。
仅交换 h 的 chunk/head 维；BF16、K=V=128、chunk=64、state_v_first、
输出 layout、strict cu_seqlens 和全部原有平台支持范围保持不变。

## 设计与追溯

- dense h `[B,C,HV,128,128]`，packed h `[1,totalC,HV,128,128]`。
- StateOffset 为 `((b*C+c)*HV+hv)*128*128`；packed c 使用 globalChunk。
- AIC 和 AIV mover 均调用同一 StateOffset；不改变 tile、缓冲、同步或模板选择。
- ACLNN、tiling、ctypes 校验同步；Stable ABI 原样透传 h，无独立旧布局分配。
- V2 前向直接把 hCompute 传给 Finalize，删除唯一剩余 head/chunk 转换及失效 helper。
- 旧融合前向 kernel、GDN 消费者、dh、h0/ht/dh0/dht 不属于本批修改。
- fast launch 使用旧融合前向的内部 Finalize，未调用本独立算子，不改变其布局。

## 验证计划与状态

- CPU：旧 FP32 标杆的等价布局映射；FP64 golden + FP32/BF16 benchmark；
  原值域 [-1,1]，不放宽 mixed_tolerance_bm；验证错误 chunk/head 映射敏感性。
- ATK：26 个 profile × 4 layout × 2 state order × 3 seed = 624 精度用例；
  10 性能用例；12 内存/确定性用例。覆盖 AIC-only 与 A5 AIV mover。
- 公共入口：两后端运行专项 FP64、重复执行和无效输入测试；原有 11 组 parity。
- 链路：FwdH 原有 24 组及 A5 KDA 保存/重算 6 组；#696 L2Norm dense/packed Example/ST。
- 平台：A2/A5 wheel 构建、安装产物核对、单算及链路；A3 编译支持另行记录。
- profiling：同 shape 前后 Finalize 与 V2 完整链路，确认 h 转置消失。
- NPU CI：对最终提交执行仓库允许的双平台 quick 门禁，记录实际权限与结果。

## 恢复后的 A2 验证结果

- wheel 与安装产物哈希核对通过；Stable ABI/ctypes 各 78 组通过。
- FwdH 24 组通过、6 个 A5-only 跳过；公开入口 parity 11 组通过。
- 原 mixed_tolerance_bm 624/624；ATK 发起执行并追加 CT Tool 0.9.1 L1
  双标杆 624/624，NPU BF16 DUT / CPU FP64 golden / CPU FP32-BF16 benchmark。
- 确定性 12/12；dense/packed KDA Example/ST 正反向精度均通过。
- 新旧 NPU 输出 12 组逐位相同；10 组 Finalize kernel 性能配对最大增加 0.31%。
- V2 profiling 中 Prepare/FwdH/Finalize 各 25 次，无转置任务。
- 插桩内存测试执行了 12 组目标 Finalize，但有 FFTS_BASE_ADDR 寄存器恢复
  warning；不能将它记为干净的 sanitizer Pass。系统 ReduceAll 有同类 warning。

## A5 恢复后的结果（2026-09-21）

- wheel 构建与安装溯源通过，62 个 kernel 文件一致。
- Stable ABI/ctypes 各 78/78；FwdH 链路 30/30；parity 11/11。
- mixed_tolerance_bm 与 ATK+CT 双标杆均 624/624，逐 case ID 核验通过。
- dense/packed Example/ST 正反向通过，新旧输出 12/12 逐位相同。
- V2 链路三阶段各 25 次，无 h 转置任务。
- 原确定性报告有 2 个 TIMEOUT；空闲卡 7 重跑 50 次循环后 12/12 通过。
- 十组单算性能已配对；首次 case 5/7 的明显增加在空闲卡两轮交替复测中
  未重现，变化范围分别为 -0.006%～+1.250%、-0.181%～+0.113%。
- 内存检测未通过：MIX case 8（B=384,HV=2,T=17）在 AIV→L1 搬运处
  报 `illegal write of size 544` 和 507015；空闲卡单例重复验证仍复现。
  旧的 ATK 汇总跳过了未找到的报告，不能据其退出码记作内存 Pass。
- A5 旧基线的独立检测对照已启动，随后 SSH 再次超时，结果待取回。
- A2 修改前基线 case 0 已复现相同 FFTS_BASE_ADDR 告警；该告警不是本次
  NT-first 引入，但仍不将检测记录写作干净 Pass。

双平台 quick CI、完整模型性能目标和 A5 内存异常仍未收尾。
本阶段仍未完成，不作为正式验收或合入结论；不修改阈值或禁用 AIV 分支规避失败。

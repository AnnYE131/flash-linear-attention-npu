# Phase6 A2/A3 内部实现

本目录只承载 `Ascend910B`（A2）和 `Ascend910_93`（A3）的 Phase6 私有实现。代码从
冻结提交 `chw@e0fb8942336efcf3b2e13d7463e63b5b5d38f341` 的独立算子路径复制，公共
`ChunkGatedDeltaRuleFwd` 入口不再通过同级算子目录获取实现。

维护约束：

- `arch22` 不包含或引用 `arch35` 文件；A5 代码不得在此目录修改。
- A2/A3 精度修复只进入本目录，并以冻结的 A2 双 500 ATK 基线回归。
- 独立算子仍可保留公开注册，但 Phase6 不依赖其源码路径或构建产物。
- 公共 ABI、输入输出顺序和 tiling 序列化布局不因内部化改变。

## 2026-09-10 同步修复

本次按照设计规则V2修复已有流水的依赖，不改变计算公式、输入域、布局、workspace大小或核间参与者。

| 位置 | 已定位问题 | 修复和生命周期 |
| --- | --- | --- |
| 公开cumsum的offsets | Scalar写入的索引随后由Gather读取，原PIPE_V不足以建立S→V依赖 | 写完索引后增加同核S_V SetFlag/WaitFlag，使用当前TPipe内的EVENT_ID0 |
| 公共BlockMmadTla的UnitFlag分支 | L0C事件列表没有初始化，但MTE1→M路径仍读取列表 | 在该分支设置l0CEventList[0]=0；原模板已经约束L0C_STAGES=1 |
| H零初态初始化 | 两个UB槽交替复用时，旧MTE3→MTE2等待没有传递给新的Vector写入 | 在Duplicate前增加同eventId的MTE2_V配对，形成旧V→MTE3→MTE2→新V的依赖链 |

公共MMAD修复还影响使用UnitFlag的其他调用者，包括独立GDN算子和KDA；未开启该分支的模板行为保持原样。
本轮A2验证范围不得解释为A5硬件验证。

开发期已完成固定FP16 case5的普通和插桩诊断构建，o/g_cumsum/A与已通过CPU FP64和六ACLNN标杆的冻结输出逐位一致。
已修复MSS解析器的错误核号，以及A2 FP16 Duplicate记录的地址截断；后者以真实高UB地址写入和正负race微探针确认。
H复用微探针在缺少MTE2→V依赖时报告WAR/WAW，加入依赖后通过；即使负例数值偶然正确，也不能据此忽略数据竞争。

完整源码构建、原有输入矩阵、16种MSS签名及长流水回归仍需以当前源码重新执行。
开发记录与原始证据位于同级旁车仓`A2SolveTri_Triton.ai-work/tasks/a2-solvetri-triton-opt/`，其中`STATE.md`为唯一当前状态。

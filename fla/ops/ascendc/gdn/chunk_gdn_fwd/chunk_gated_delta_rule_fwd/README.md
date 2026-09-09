# ChunkGatedDeltaRuleFwd

## 功能

`ChunkGatedDeltaRuleFwd` 实现 Gated Delta Rule 的分块前向计算。A5 上 `useExp2=true`
且 `useQkL2norm=true`
时依次调度 `ChunkGatedDeltaRuleFwdPrepare`、`ChunkGatedDeltaRuleFwdH` 和 `ChunkFwdO`；
其他组合继续使用原 Phase6 kernel。当前实现支持定长和变长序列、GVA、可选初始状态
以及可选最终状态输出。

用于精度对比的公开算子链依次由以下算子组成：

1. `ChunkLocalCumsum`（`chunk_local_cumsum`）；
2. `ChunkScaledDotKkt`（`chunk_scaled_dot_kkt`）；
3. `SolveTri`（`solve_tri`）；
4. `RecomputeWUFwd`（`recompute_w_u_fwd`）；
5. `ChunkFwdH`（`chunk_fwd_h`，固定 `use_exp2=true`）；
6. `ChunkFwdO`（`chunk_fwd_o`）。

融合 kernel 内部实现上述等价计算阶段，不调用或链接这些公开算子的 ACLNN 实现。
公开算子链只作为 ATK 精度标杆使用。

## A5 Phase6 的 WU 流水

自然指数 Phase6 的 A5 私有 WU 实现按同一 chunk/head 任务依次生产 `Vb` 与
`KbgExp`，Cube 紧接着计算 `U = A @ Vb`、`W = A @ KbgExp`。
两次矩阵乘复用驻留 L1 的 A；保持 FP32 向量运算与原输入 dtype 的 RINT 写出。
任务归属继续使用系数阶段的连续分片，定长/变长和尾块仍由共同任务解码器处理。

每个 AIC 组使用 8 个 GM 环形槽；对应的两个 AIV 子核按任务轮流独占整块数据的写入，
非 owner 子核只发送配平的 ready 通知。两个子核均按所有任务的相同次序消费 free credit。
槽号为组内任务序号模 8；Vb 区在前，KbgExp 区在后，工作区总量为
`sizeof(input) * AIC数量 * 8 * chunkSize * (V + K)` 字节。
该固定槽数减小长序列临时空间，但小形状的占用可能高于原整张量工作区，内存验收需单独比较。

| 依赖 | 通知与等待 | 槽位生命周期 |
| --- | --- | --- |
| Vb 写入→U 读取 | AIV MTE3 发布 flag 3，AIC 等待 | 每任务由 owner 写入；两个子核均通知 |
| KbgExp 写入→W 读取 | AIV MTE3 发布 flag 4，AIC 等待 | 与 Vb 使用相同任务次序 |
| GM 读取完成→覆盖同槽 | AIC 在最后一次 MTE1 消费后用 flag 5 广播，两个 AIV 等待 | 前 8 个任务使用空槽；回绕前等待，结束时排空剩余 credit |

向量输入、输出及 gate 队列双缓冲，Vb/KbgExp 复用同一套 UB 队列。
host 从 chunkSize 向下折半选择公共行块，至少 8 行，并保留 16 KiB UB 余量；
两路使用一致行块，避免队列复用时越界。Cube 使用单缓冲 L0C 的 UnitFlag 形式。

Solve→WU 的既有发布和全 AIC 完成等待、WU→H 的入口同步保持原样。
进入 WU 前已消费 Solve 使用的 flag 4/5，WU 结束后消费其全部 credit；阶段间复用编号
不允许重叠在途代次。此改动限于 A5 私有实现，A2/A3 和 Prepare→H→O 路径保持既有实现。

## 输入

令 `B` 为物理 batch size，`Hk` 为 q/k 头数，`Hv` 为 v 头数，`T` 为 token 数，
`K=128`，`V` 为 value 维度，`N` 为逻辑序列数。

| 名称 | 必选性 | Shape/Dtype | 说明 |
| --- | --- | --- | --- |
| `q` | 必选 | 由 `layout` 决定；FP16/BF16 | Query |
| `k` | 必选 | 与 q 同 shape/dtype | Key |
| `v` | 必选 | 由 `layout` 决定；与 q 同 dtype | Value |
| `g` | 必选 | `[B,T,Hv]`；FP32 | 门控值，固定为 sequence-major |
| `beta` | 必选 | 与 g 同 shape；FP32 或与 q 同 dtype | Delta 系数 |
| `aLogOptional` | 当前未支持 | - | 扩展接口预留，必须为空 |
| `dtBiasOptional` | 当前未支持 | - | 扩展接口预留，必须为空 |
| `initialStateOptional` | 可选 | `stateVFirst=false` 时为 `[N,Hv,K,V]`，否则为 `[N,Hv,V,K]`；FP32 或与 q 同 dtype | 初始状态 |
| `cuSeqlensOptional` | 可选 | `[N+1]`；INT64 | 变长序列累计长度，需与 `chunkIndicesOptional` 同时提供 |
| `chunkIndicesOptional` | 可选 | `[2*Nc]`；INT64 | canonical sequence-major chunk 索引 |

四种 QKV 布局均使用四维输入：`BNSD/NTD=[B,H,T,D]`，
`BSND/TND=[B,T,H,D]`。`Hv` 必须能被 `Hk` 整除。变长模式使用物理 `B=1`，`cuSeqlensOptional` 必须从 0 开始、
以 `T` 结束且单调不降。

## 输出

| 名称 | 必选性 | Shape/Dtype | 说明 |
| --- | --- | --- | --- |
| `oOut` | 必选 | `[B,T,Hv,V]`（BSND）；与 q 同 dtype | 前向输出固定使用 sequence-major 布局 |
| `finalStateOutOptional` | 可选 | 末两维由 `stateVFirst` 决定；与初始状态同 dtype，无初始状态时为 FP32 | 是否为空直接决定是否计算并输出最终状态 |
| `gCumsumOutOptional` | 可选 | 与 g 同 shape；FP32 | chunk 内门控累加结果；为空时使用内部临时张量 |
| `aOutOptional` | 可选 | `[B,Hv,T,chunkSize]`；与 q 同 dtype | 系数矩阵；为空时使用内部临时张量 |
| `qHatOutOptional`、`kHatOutOptional` | A5 `useExp2=true` 可选 | 与 q/k 相同 | L2Norm 结果 |
| `qRstdOutOptional`、`kRstdOutOptional` | A5 `useExp2=true` 可选 | q/k layout 去掉最后一维；FP32 | L2Norm rstd |
| `betaEffOutOptional` | A5 `useExp2=true` 可选 | 与 beta 同 shape；FP32 | 非空时启用并输出 beta sigmoid |
| `hOutOptional` | A5 `useExp2=true` 可选 | `stateVFirst=false` 时末两维为 `[K,V]`，否则为 `[V,K]`；与 q 同 dtype | 分块状态 |

Python ctypes 入口通过 `disable_recompute=False` 选择训练输出，返回 `gCumsum` 和 `A`；
设为 `True` 选择推理输出，仍返回四元组，但后两项为 `None`，底层公共输出指针也为空。

## 属性

| 名称 | 当前支持范围 | 说明 |
| --- | --- | --- |
| `layout` | A5 `useExp2=true` 支持 `BNSD/BSND/NTD/TND`；其他路径保持原有支持范围 | q/k/v 的输入布局；BSND/TND 输入在拼接路径内转为 head-first，o 固定输出 BSND |
| `scale` | 通常为 `K**-0.5` | Query 缩放因子 |
| `chunkSize` | `64`、`128` | 分块大小 |
| `useExp2` | A5 支持 `true`；其他路径为 `false` | A5 true 时走三小算子拼接路径 |
| `useQkL2norm` | `true/false` | 仅与 `useExp2=true` 同时成立时走三小算子拼接路径；false 时走原 Phase6 kernel |
| `allowNegEigval` | A5 `useExp2=true` 支持 | 为 true 时必须提供 `betaEffOutOptional` |
| `stateVFirst` | A5 三小算子拼接路径支持 `true/false` | 控制初始状态、分块状态和最终状态的末两维采用 `[V,K]` 或 `[K,V]` |

当前未实现的扩展组合会返回参数错误，不会静默忽略。

## 支持范围

- A2（`ascend910b`）、A3（`ascend910_93`）、A5（`ascend950`）。
- 原 Phase6 路径支持 FP16、BF16，`K=128`、`V=128/256`、`chunkSize=64/128`。
- A5 `useExp2=true` 路径支持 BF16、`K=V=128`、`chunkSize=64`、`Hv/Hk in {1,2,3,4}`。
- 支持 MHA、GVA、定长和变长序列。
- A2/A3 使用 `arch22` 私有实现，A5 使用 `arch35` 私有实现；两套架构代码隔离维护。

## 验证

ATK 用例和执行说明位于
[`tests/atk/chunk_gated_delta_rule_fwd`](../../../../../../tests/atk/chunk_gated_delta_rule_fwd/README.md)。
精度双标杆分别使用融合算子、上述公开算子链和 FP64 recurrence，并在相同冻结输入上比较结果。

ACLNN ABI 合同可通过以下命令检查：

```bash
python3 tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py
```


## A5 输出阶段流水

Phase6的A5私有FwdO实现使用RegBase epilogue和分段MMAD流水。QK与QH通过两个L1槽复用Q，QH预取H时允许QK完成剩余计算；AttnV使用独立的L1区域和事件，在QH计算期间预取V，并在掩码结果发布后读取AttnMask。

Cube1/2共用的L1区域最大到192KiB，Cube3从192KiB开始使用独立区域，V256时最大到384KiB。L0计算窗口依次排空，GM中间结果的ping-pong槽在Vec2完成读取后才归还。尾块按实际行数写回，零行AIV仍配平跨核通知。

该流水使用私有实现及原有DTYPE_Q分派，通用路径保留varlen的保守同步。A2/A3私有实现、公开接口和Prepare拼接路径沿用各自实现。

## A5 模型同步策略

Phase6在BF16 qkv、BF16/FP32初态、`B=1,Hk=16,Hv=32,T=11274,K=V=128,C=64`、单变长序列、177个chunk且输出最终状态时选择内部key301。其他合法输入继续使用key1/key2；公开参数和tiling结构体不增加字段。

key301将Solve64的任务解码与KKT/WU的连续head-major区间对齐，再启用KKT→Solve和Solve→WU的组内交接，以及FwdO通知聚合。cumsum/score发布、H初始化和H→O交接仍保留相应的全局发布。Solve的尾块可能由AIV写回，AIC必须排空包含尾块通知等待的全部流水后才能释放WU。

H阶段按角色排空WU生产流水，Cube1只排空FIX，状态写回使用已有MTE3事件。仅该策略的BF16初态、V128路径将状态更新行块由16扩大为64；FP32初态保持16行。主干H尾块的MTE3_V保护仍保留。key301沿用DTYPE_Q编译分派，仅增加匹配BF16输入和初态的特化。

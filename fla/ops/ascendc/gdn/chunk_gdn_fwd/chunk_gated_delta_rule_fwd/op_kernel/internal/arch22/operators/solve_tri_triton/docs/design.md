# A2 私有 Solve 迁移设计

方案设计规则版本：V2。内部合同见 [api.md](api.md)。

## 1. 目标

将 f2fbb243 的已测分层 FP32 Solve 迁入 chw_a2_a5 的私有 A2 路径；
KKT、WU、H、O、公开接口和 DTYPE_Q 分发不变。不是新写一套数学算法。
迁移前基线 fa381c6f 已完成 500×双模式×三实际种子；旧来源精度不能替代新接入验收。

首版只验证接入：保留 BT64 varlen 的 BHT→TND→BHT，保留独立 FP32 数据版本。
不移植旧 fast ABC，不把旧整体提速作为本组件收益。
A2 专用宏仅用 COMPUTE_UNIT Ascend910B1 添加；其他 SoC 仍编译旧 Solve。
host 同时要求 SoC=ASCEND910B、NpuArch=DAV_2201 才增加资源。

模型配置：360 q/k/v=[32,16,1082,128]，无 state；
农行代理 q/k=[1,4,8088,128]、v=[1,8,8088,128]，cu=[0,4044,8088]，
原初始/最终 state 配置保持。两者 BF16、BT64、K=V=128，分别测训练/推理。
代理边界不是实际农行边界。目标为相对本次主仓基线稳定改善，不能仅看汇总中位数。
同设备/输入/预热/采样交替 A/B；仅用 msprof 目标 megakernel 的 Task Duration(us) 判定，
Event 仅观察完整调用链。详细测量入口在 validation 中固定；当前没有新版性能结论。

可达性：满块 S→2S 每 work 执行两次 S³ GEMM，合计 4S³ FLOP；
16 行叶子保留原 FP32 Vector 递推。两个模型有足够多的 32/64 行任务，
可摊薄 16-task ready 开销；但 FP32 中间读写及额外 barrier 是代价。
全量 X 转换读 2RBT、写 4RBT 字节；D16/D32（BT128另 D64）都保留 GM 版本。
未有本基线的分阶段目标 profile，不能给出可信的毫秒下界或保证收益；
先完成正确性接入，再用同条件 profile 判断保留与否。

## 2. Stage 完整详设

### 公共任务和地址

C 为本次 megakernel 实际 AIC blockDim，CG=1，Hv 独立求解；Solve 不再读取 q/k，
因此无 head 共享读收益。Hv→Hk 仍由前序 KKT 的 hv/(Hv/Hk) 完成。
32/64/128 阶段分别取 span，Nwork=dense 的 B×Hv×ceil(T/span)，
或 varlen 的 Hv×sum(ceil(seqLen/span))，禁止从 BT 任务简单倍乘。

每核本地 index i 对应全局 task=c+i×C；两个 paired AIV 与 AIC 使用同一映射。
分组 16/8 是每核连续 work 的流水批量，不是 CG，不跨 head 合并矩阵。
dense task 展开为 chunk→B→Hv；varlen 为 sequence→chunk→Hv。
每个 task 的两 AIV 分别写上半行/下半行，输出互斥。无效半片仍消费和释放通知。

Solve 活跃组为 min(C,Nwork)。整 megakernel blockDim 和 H 的有依赖调度保持原状；
Nwork<C 时其余物理组只参加 mixed 边界。不能单独缩小物理 blockDim 破坏前后阶段。
这属于既有融合调用的兼容约束，不声称改造了完整 mega 的 V2 分核。
CG=2/3/4 会同时改变缓冲/flag 所有权，本轮不引入；无独立收益证据不增加模板维度。
旧来源是连续均分，本候选是 grid-stride；改变的是地址所有权顺序，必须重新验证。

以 360、C=20 代入：N32=17408，前8核871个任务、其余870；
N64=8704，前4核436、其余435。农行代理 N32=2032、N64=1024。
core 数来自运行时而非硬编码20。空序列自然跳过；所有有效行只属于一个 task。

设局部 w=(b,h,t,base,length)，row 为序列内行：
headFirst=1：pos=((b×Hv+h)×T+base+row)×width；
headFirst=0：pos=((b×T+base+row)×Hv+h)×width。
行 stride 分别为 width 和 Hv×width。X 子块列偏移 t%BT。

| Stage | 单元与公式 | 空间 |
|---|---|---|
| 0 | Vector：raw→X FP32 | UB 96 KiB |
| 1 | Vector：每个16行叶子 D16=(I+M16)^(-1) | UB 与 Stage4 共峰值84 KiB |
| 2 | Cube：P=D1 @ M10，累加/写出FP32 | L1 64 KiB；scratch临时槽 |
| 3 | Cube：C10=P @ D0，累加/写出FP32 | 同一L1，scratch输出ring |
| 4 | Vector：组合 [[D0,0],[-C10,D1]] | BT32 2 KiB，BT64 12 KiB，BT128 48 KiB |
| 5 | 重复2/3/4，先32→64，必要时64→128 | 分层间mixed barrier |

Stage1 与 16→32 的 Stage2/3/4 保持旧来源一批提前流水。
第一级合并输出 D32 FP32；第二级在 BT64 写最终低精度，否则写 D64 FP32；
第三级只在 BT128 写最终低精度。最终 CAST_RINT，不启用 HF32。
没有新增倒数/除法，单位三角对角固定为1。
CPU inv(I+M) 对应叶子递推与分块恒等式，不是与自身低精度路径互相比较。

数值风险：输入合同要求严格下三角，叶子先否定 M，按旧固定 Add 树递推，
对角最后写1；只读取有效 n×n，先零填充 UB，再进行 vector 运算。
有效域外不参与 Cube M/N/K，输出尾列由零初始化保证。
任意无界 M 的逆可能超出 FP32/输出 dtype；沿用父算子有限值要求和已校准 ATK 域，
不裁剪数据、不放宽阈值、不把有限输入等同于逆一定有限。
两种低精度输入转 FP32 精确，所有中间保持 FP32，最终 cast 风险由三标杆/有限值门禁约束。

### Stage0：Vector，完整低精度转 FP32

每 AIV 按 16384 元素一段、全局 AIV grid-stride 处理。
BT 保证末段为64元素倍数；连续 DataCopy，不读取额外 padding。
UB（字节）：输入 [0,32768)，输出 [32768,98304)，空闲 [98304,196608)。
MTE2→V 事件0，Cast→MTE3事件0，PIPE_ALL 完成后复用两区域。
此 Stage 结束后 mixed barrier，X 到后续所有组可见。

### Stage1/4：Vector，叶子与合并

叶子 L=32，每 AIV 为32份任务分别计算16×16矩阵，不同半片地址不重叠。
实际 TPipe 先分配合并 f，再分配叶子 a/prod/row/coeff：
UB字节 [0,2048) f；[2048,34816) a；[34816,67584) prod；
[67584,69632) row；[69632,86016) coeff；[86016,196608) 空闲。
a 经 MTE2 写入、Vector 递推、MTE3 写 D16，MTE3_V 后允许下一 Produce 复用。
prod/row/coeff 仅当前 Produce 活跃；f 在产生未来叶子时不保留有效上一结果。
A2 使用原 Vector API，不使用 A5 VF 特性。

32→64：UB [0,8192) f，[8192,12288) cast输出（FP32输出时无此段）。
64→128：UB [0,32768) f，[32768,49152) cast输出。
f先清零，读本半片 D 和（仅下半）C10，经取负/可选cast写完整2S列，
MTE3_V确认写完后再释放bank。源中间均独立GM版本，无覆盖前一阶段D的别名。

### Stage2/3：Cube，分组双 GEMM

CATLASS MmadPingpong<AtlasA2,true,false>，Tile=64³，FP32 row-major。
目标依赖41bf90da：L1 A双份 [0,16384)、[16384,32768)；
B双份 [32768,49152)、[49152,65536)，[65536,524288) 空闲。
L0A/L0B各 [0,16384)、[16384,32768)，其余至65536空闲；
L0C [0,16384)，其余至131072空闲。FP32累加，Fixpipe写 GM，不写 UB。
两次 GEMM 复用同一 CATLASS资源；MTE1_MTE2/M_MTE1各类型事件0..3，
构造初始化，析构等待末轮；AIC资源生命周期不跨分层mixed边界。

16→32 的 Group=16，32→64 的 Group=8：先整个group计算P到临时槽，
PIPE_ALL完成GM RAW，再计算P@D0到输出ring，PIPE_ALL保护临时槽WAR。
64→128逐task执行两次GEMM，第一次后PIPE_ALL，最终Fixpipe ready保护读者。

GM relay 是本次已测 A2 组件的兼容路径：现有 CATLASS copy-out 写 GM，
批量P与跨引擎ring的生命期超出单次MM调用。改为L1 resident或Fixpipe→UB
会改变copy策略、跨单元可见性和同步协议，本次不同时重写。
相对V2 R04/R07/R10的驻留优先方案，这是明确保留的迁移代价，
不是容量已证明不足，也不宣称满足全部新算子最优驻留规则。

### Workspace 与同步闭环

host 保留原score/a/g和全部suffix区域；以下新资源各按512字节对齐，均独立：
X=4RBT，D16=4R16，D32=4R32，BT128再加D64=4R64。
每层scratch相对基址（单位FP32元素）：临时槽 [0,C×TEMP×E)，
结果ring [C×TEMP×E,C×(TEMP+32)×E)，E=max(S,32)²。
core临时地址 core×TEMP×E；ring地址 (core×32+index%32)×E。
S16/S32 TEMP16，S64 TEMP2；总scratch为每核192/544 KiB。
其中64→128只用前一个临时槽，第二个保持旧分配；不是额外并行证据。
各层完全drain并mixed barrier后复用scratch；X/D保留至Solve结束。

满work各层读 D0/D1用于两次GEMM共2S² FP32，M10 S²；
P写读各S²，C10写一次、下半AIV读一次；组合再读D0/D1共2S²，
输出4S²（FP32中间或低精度最终）。D16另有输入/输出各256×4/leaf，
X转存读写如上。GM数据重读已明列，不隐藏为零开销。
BT64 varlen额外两次staging，每次读写各2RBT字节，生命周期沿用原score区。
20核计划额外workspace：360 250806272字节；代理31608832字节，须以真实tiling核验。

同步伪代码（每个core group，两个AIV必须同序参与）：

```text
所有物理核：mixed barrier；AIV转X；mixed barrier
AIV：先Produce本地32-task叶子版本0，MTE3发leaf_ready[0]
每个32-task版本v：
  AIV提前Produce v+1，发leaf_ready[(v+1)%2]，D16按task唯一地址
  AIC每32任务等待leaf_ready[v%2]（两个AIV发布）
  AIC每16任务批b：若b>=2先等free[b%2]
    计算GEMM1、barrier、GEMM2、barrier；FIX发ready[b%2]
  两AIV等ready[b%2]，逐task组合/写回/MTE3_V；MTE3发free[b%2]
AIC最终等最后min(2,batches)个free；CATLASS析构drain
所有物理核mixed barrier；其余层重复16-task ready/free协议
所有物理核最终mixed barrier；若TND staging则转回BNSD后再次mixed barrier
再进入原WU/H/O；不执行旧PHASE6_SOLVE_DONE_FLAG
```

leaf_ready物理flag0/1；free物理2/3；ready物理4/5。
首次两个输出bank可直接用；第三批覆盖前必须等两AIV的free；
末尾不足16仍发一次ready/free；无task不发不等局部flag，但参与全局barrier。
leaf版本的GM不复用，AIV最多提前一版本；不能把leaf flag0/1误当GM ring释放。
本地AIV各HardEvent类型的事件0独立；KKT退出、Solve层退出均完成相关数据和事件。
以上是设计推导；真实flag语义、复用轨迹、MSS和精度需按validation验证，不以计数审计代替。

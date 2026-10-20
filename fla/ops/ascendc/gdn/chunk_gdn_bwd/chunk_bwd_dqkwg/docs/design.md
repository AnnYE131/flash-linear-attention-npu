# P154：DQ 完整 HK 组均衡推广到合法 varlen

方案规则版本 V2；唯一父为 P152 `816ff1a8c938e201bd338f063cdcedbe1a4337d5`，其稳定起点为 P144 `9c9f91cf5e2800e58646c1c87c585aa7a0cbc200`。本文件只补充 DQ host 资格差异；公开接口、CPU 标杆、设备数学和支持域沿用父版，不混入独立 P153 四头 WY。Finalize 目录旧 P117 文档保留为历史，不作为本候选现状。唯一 HANDOFF 仍在原工程根目录。

## 选择规则与整数边界

保留父选择 `P0=min(HK,max(1,C/N))`，N 为最终 `B*numChunks`、C 为原清洗后 AIC 数。正式 tiling 的非950外层 guard 保持原样，950完整父路；现有 ascend910b fast-launch processor 使用同一个无 SDK host helper。只把两入口的 allowBalance 扩为 `isVarLen!=0 || T%BT==0`；其余资格均保持 P152：N≥C，且父 `DqkwgUseU32Schedule(...,C,P0)` 成立。

HK约数配对枚举、每区偶数HK、`W(p)=ceil(N*p/C)*(HV/p)`、INT64乘法前界定、除余ceil、U64乘法保护及按 `(W,p)` 取最小且严格改善才改P，全部评分正文逐字父。p>1仍原 WIDE64；不扩大父 U32 loop/bos/index 或 metadata 地址域，不新增字段、元数据读取、队列或事件。

## varlen 语义与资源

varlen仍要求B1，N取最终chunk_indices对数。设备先用 `loop=task/P` 找原扁平ordinal，再解 `(seq,localChunk)` 和实际 `bos/real_BT`；h/dh状态仍按同一ordinal寻址。完整HK组的R个HV仍在同task/同owner按原h升序完成，合法尾块保留原有效长度/两半路径。不得把T%BT当无尾块证明，不重排chunk_indices，不按测试名称或固定shape选择。

N≥C时ring仍C；workspace字节/偏移、blockDim、UB/L1/L0、FP32/Cast/归约和每task原ready/clear/credit/lease均不变。Cube/Vector/common/schedule/private/WY/SDK逐字P152。task增p倍，原协议与WIDE64成本也增加；计分只表示分配头量，不是加权执行时间或提速上界。实际20核下N256/HK16/HV128选P8、峰头量1664→1648仅为整数例，尾块可合法分区不等于其成本均匀。

P152 C01原同轮Task14750.3547us，对P14418356.6629us，普通Event15158.0458/18772.3087，候选CV6.3214%，共享卡7；只是P152已闭首项证据，未据此认定具体机理、全20或精度通过。本P154尚未测量，不把两个候选收益相加。

参考 CHUNK_INDEPENDENT V1（来源 commit `a78efa9512dfa64c20b2a58a75e84e88a054d694`）仅采用完整输入下独立task与资源闭环的核对方法；不引入该参考的Stage、tile、buffer或runtime。

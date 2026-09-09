# A2 私有 Solve 接口

本组件只供 arch22 megakernel 内部调用，不新增 Python、aclnn 或公开算子入口。
公开合同以父算子 README 和原接口实现为准，本文件只定义内部边界。

- 目标：Ascend910B / DAV_2201，MIX_AIC_1_2，CANN 9.1.0-beta.1，auto-sync=off。
- 输入 raw：KKT 已写好的严格下三角 M，对角和上三角为零；输出为 (I+M)^(-1)。
- 输入和输出为同一种 FP16/BF16；X、D16、D32、D64、scratch 为 FP32。
- BT 为 64/128，R=B×Hv×T。所有地址以字节为单位，由 host 私有 trailer 给出。
- headFirst=1 为 [B,Hv,T,width]；headFirst=0 为 [B,T,Hv,width]。
  父入口始终 BNSD；首版仅 BT64 varlen 在已有 staging 内采用 headFirst=0。
- dense：B/Hv/T>0；varlen：B=1，cuSeqlens 为 INT64 的单调不降完整分区。
  sequences 为序列数；零长度序列无 task，非空序列每个子阶段独立向上取整。
- 最终输出每个有效行的 BT 列完整写回；尾部无效列为零，不越过实际 T。
- 所有启动的 AIC 与 paired AIV 都必须调用 Run，包括无本地 Solve task 的核；
  不能在外部只由部分核调用，也不能用旧 AIC/FIX flag 代替末尾 mixed barrier。
- 训练/推理共用同一内部 Solve；推理是否公开 A/G 由现有父入口决定。
- caller 必须保证 raw 与 X/D/output 不重叠；仅已结束阶段的 scratch 可跨阶段复用。
  BT64 的 D64 offset 可等于 D32，因为该分支不访问 D64。

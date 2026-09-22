# ChunkFwdO：NT-first h 契约

本次仅修改 h 的 chunk/head 轴顺序。函数签名、dtype、输出布局和平台支持范围沿用
[算子说明](../README.md)。

- dense h：`[B,numChunks,HV,K,V]`；带 cu_seqlens 的 packed h：`[totalChunks,HV,K,V]`。
- `stateVFirst=true` 仅交换末两维为 `[V,K]`，原有平台限制不变。
- q/k/v/g 的布局保持不变；h 必须来自匹配版本的生产者。
- 共享 ChunkFwdH 输出可直接传入；旧独立 ChunkGatedDeltaRuleFwdH 输出需先
  `h.transpose(1,2).contiguous()`，varlen 再 `squeeze(0)`；该临时适配随旧 FwdH 迁移移除。
  不根据维度大小自动猜测布局，HV=numChunks 时也按 NT-first 解读。
- GDN 组合前向 A5 prepare 路径支持 hOutOptional，遵循同一契约；旧融合路径的支持域不变。

## P2 实现说明（设备验证待执行）

h/dh 的统一目标、packed rank 与末维顺序见 [迁移契约](../../../../../../../docs/architecture/h-dh-nt-first-contract.md)。
ACLNN 校验 packed rank-4，连续化后用 reshape 补 B=1 视图供原 rank-5 tiling 使用。
该视图不交换 NT/HV，也不增加状态数据搬运。内部 L0 和 fast launch 仍使用 rank-5 descriptor。

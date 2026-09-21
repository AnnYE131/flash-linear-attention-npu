# ChunkFwdO：NT-first h 契约

本次仅修改 h 的 chunk/head 轴顺序。函数签名、dtype、输出布局和平台支持范围沿用
[算子说明](../README.md)。

- dense h：`[B,numChunks,HV,K,V]`；packed h：`[1,totalChunks,HV,K,V]`。
- `stateVFirst=true` 仅交换末两维为 `[V,K]`，原有平台限制不变。
- q/k/v/g 的布局保持不变；h 必须来自匹配版本的生产者。
- 共享 ChunkFwdH 输出可直接传入；旧独立 ChunkGatedDeltaRuleFwdH 输出需先
  `h.transpose(1,2).contiguous()`。不根据维度大小自动猜测布局，HV=numChunks 时也按 NT-first 解读。
- GDN 组合前向的 `hOutOptional` 仍为预留未开放，本次不扩展其支持范围。

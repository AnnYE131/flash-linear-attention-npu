# KDA BwdFinalize 状态接口

完整签名与 A5 v1 限制见 [README](../README.md)。
h/dh 均为 dense `[B,NT,H,K,V]` 或 packed `[totalNT,H,K,V]`。
packed token 为原有 rank-3 `[H,T,D]`，不增加 batch 维；内部按相同 NT-first 地址访问。
该版本仍要求 K=V=128、chunk_size=64、state_v_first=false，其他属性限制保持原契约。

cu_seqlens 从 0 到 T 非递减，chunk_indices 须完整且按 sequence-major 排列；
允许空序列条目，但总 token 数与总 chunk 数须为正。ACLNN 校验 metadata 值，tiling 校验两种状态的 rank、容量、head 与末维。
无需在 Dhu 与 Finalize 间换轴。本轮源码已迁移，设备验收待 P6。

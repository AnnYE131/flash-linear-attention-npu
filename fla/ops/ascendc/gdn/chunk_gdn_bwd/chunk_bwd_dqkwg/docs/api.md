# dqkwg 状态接口

完整签名、dtype 与支持域见 [README](../README.md)。
h/dh 均为 dense `[B,NT,HV,K,V]` 或 packed `[totalNT,HV,K,V]`；该入口仍只支持 K-first。
packed token 保持 B=1；cu_seqlens 从 0 到 T 非递减，chunk_indices 须覆盖所有 chunk，
按 sequence-major 排列，允许空序列条目，但总 token 数与总 chunk 数须为正。NT 是各序列 chunk 数之和。
host 校验 h/dh 的 rank、NT、HV、K/V；metadata 校验沿用原实现。不自动识别旧 head-first 数据。
NT=HV 时 shape 无法辨别内容，调用者必须提供新布局。

ACLNN 和 fast launch 只给 packed 状态补 B=1 的内部视图，数据不换轴。
两个 tiling 实现均检查内部 `[B,NT,HV,K,V]`。本轮源码已迁移，设备验收待 P6。

# Dhu 状态接口

签名、输入、dtype 和支持范围见 [README](../README.md)。
dh 为 dense `[B,NT,HV,K,V]` 或 packed `[totalNT,HV,K,V]`。
`state_v_first=true` 同时交换 dh 的 K/V 末维和实际存储，Cube 读回遵循相同布局。
h0/dht/dh0 没有 chunk 轴，保持 `[N,HV,K,V]` 或 `[N,HV,V,K]`。
没有 h0 时 dh0 为空。

packed token 保持 B=1；cu_seqlens 从 0 到 T 非递减，允许空序列条目，但总 token 数与总 chunk 数须为正；chunk_indices
须完整、按 sequence-major 排列。NT 与 metadata、输出 descriptor 必须一致。
错误 rank、容量、head、尾维在 host 拒绝；metadata 校验沿用原实现。

已知原有缺口：设备入口忽略 dht；非零末态梯度仍待修复，不计入已通过支持域。
本轮源码与 CPU 标杆已迁移，设备验收待 P6。

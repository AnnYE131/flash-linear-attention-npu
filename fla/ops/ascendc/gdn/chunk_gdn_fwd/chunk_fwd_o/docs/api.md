# FwdO 状态接口

完整签名、dtype、输出 layout 与平台支持域见 [README](../README.md)。
h 为 dense `[B,NT,HV,K,V]` 或 packed `[totalNT,HV,K,V]`；支持 state_v_first 的路径交换末两维。
token 输入保持原有布局，packed 使用 B=1。NT 是每条序列 chunk 数的总和。
cu_seqlens 从 0 到 T 非递减，允许空序列条目，但总 token 数与总 chunk 数须为正；chunk_indices（底层名 chunk_offsets）
为完整、按 sequence-major 排列的二元组列表。cu 和 indices 必须成对提供。

ACLNN 校验实际 metadata 值及 h 的精确容量；共享 tiling 校验内部 descriptor 的 NT。
fast launch 采用相同公开 packed rank-4，内部仅补 B=1 视图。
错误 rank、容量、head、尾维或 metadata 必须在 kernel 启动前拒绝。
新增检查不扩展 dtype、末维布局或平台支持域。设备验收待 P6。

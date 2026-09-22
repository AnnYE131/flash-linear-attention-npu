# ChunkGatedDeltaRuleFwdH 状态接口

公开签名、dtype、gate 与平台支持范围见 [README](../README.md)。本次只统一状态存储。

| 输出 | dense | packed（提供 cu_seqlens） |
| --- | --- | --- |
| h | `[B,NT,HV,K,V]` | `[totalNT,HV,K,V]` |
| final_state | `[B,HV,K,V]` | `[N,HV,K,V]` |

`state_v_first=true` 交换末两维及真实存储。h 是每个 chunk 开始前的状态；final_state
是每条序列结束后的状态。v_new 与 token 输入布局保持原契约。
NT 为 ceil(T/chunk_size)，totalNT 为各序列 chunk 数之和，不能按总 token 数做一次 ceil。
packed token 使用 B=1；cu_seqlens 从 0 到 T，严格递增，不支持空序列。
chunk_indices 提供时须是完整、无重复的 sequence-major `(sequence, local_chunk)` 对。
输出 rank、NT、HV、K/V 必须精确匹配；错误状态容量在 ACLNN host 拒绝，metadata 校验沿用原实现。
旧 head-first 数据不能仅 reshape，也不根据相等的 NT/HV 维度自动识别。

两种 Python 后端使用同一契约；fast launch 不新增 state_v_first 能力。
当前源码已迁移，设备编译与验收待 P6。

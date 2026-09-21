# ChunkFwdO：h 布局改动

方案设计规则版本：V2。接口差异见 [api.md](api.md)。

仅调整 h 的 GM 基址：dense 为 `((b*numChunks+chunk)*HV+hv)*K*V`，
packed 为 `(globalChunk*HV+hv)*K*V`。A2/A3 及 A5 兼容 scheduler、A5 优化路径
使用同一轴顺序。矩阵内部的 K/V 排列继续由 stateVFirst 决定。

QH 与块内 QK/AV 的数学计算、累加类型、tiling 策略、workspace、流水和同步全部保持。
GDN 组合前向直接连接共享 FwdH 与 FwdO，删除原 head/chunk 转置。
融合 kernel 内部副本和旧独立 FwdH 保持原实现；旧独立 FwdH 的 Python 调用链在
FwdO 输入边界做显式布局适配。公开 runtime 已原样透传 h，无需修改。

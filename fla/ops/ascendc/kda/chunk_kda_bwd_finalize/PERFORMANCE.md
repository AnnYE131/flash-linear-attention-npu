# KDA 反向性能记录

## 测量配置

2026-09-18，Ascend950PR_9579，物理卡 2，Stable ABI 优化分支。
B=1、H=96、T=8192、K=V=128、chunk_size=64，
disable_recompute=True、lower_bound=-1；Token/beta 为 BF16，Gate 参数为 FP32。

两组使用相同的合成前向缓存与归一化 Q/K，仅切换是否传入 rstd。
每组预热 10 次后采样 20 次；msprof 开启 task-time，
关闭 ai-core、runtime-api 和 ascendcl 采集。

## op_summary 耗时

单位：μs；统计 Task Duration，排除预热。

| Kernel | 无归一化反向：均值 / 中位数 | 融合归一化反向：均值 / 中位数 |
|---|---:|---:|
| Prepare | 1111.544 / 1110.983 | 1109.413 / 1108.890 |
| Dhu | 1467.921 / 1446.978 | 1487.994 / 1488.054 |
| Finalize | 7024.338 / 6769.528 | 7603.624 / 7579.844 |
| 每次调用三段合计 | 9603.804 / 9323.068 | 10201.030 / 10184.221 |

归一化反向融合在 Finalize 内，两组均启动三个 kernel。
三段合计均值增加约 0.597 ms（6.2%）。合计范围分别为 9.269–10.493 ms
和 9.629–10.913 ms，未锁频，结果存在波动；该表不代表端到端模型耗时或精度验证。

原始 CSV、采样脚本与汇总保存在 A5 私有账户：
`/home/Ensley/pr597-stable-20260918-pctOTJ/msprof-h96t8192/`。
对应构建与验证范围见 [验证记录](../chunk_kda_bwd/docs/optimized_validation.md)。

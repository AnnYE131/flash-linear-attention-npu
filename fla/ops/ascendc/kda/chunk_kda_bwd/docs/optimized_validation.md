# KDA 反向优化验证

## 本次验证

2026-09-18 在 A5 私有环境完成，基于 main 的 PR #597 代码。
v26.9.1 的 PR #611 移植相同改动，未单独重跑设备验证。

| 项目 | 结果 |
|---|---|
| 裁剪构建 | FLA_NPU_OPS=chunk_kda_bwd，V1/V2 符号及依赖 kernel 均包含 |
| Stable ABI 与 ctypes | 各 1 个原路径、7 个优化场景通过，输出逐位一致，无回退 |
| 非法参数 | 2 个组合在启动前拒绝 |
| 离线门禁 | A5：29/29；本地：28 通过、1 个构建哈希检查跳过 |

优化场景覆盖：定长 64、单 Token、定长尾块 65 加 rstd、packed 128、
含空序列的 packed [0,1,1,65] 加 rstd，以及定长/packed 重计算。
两条后端输出一致只验证调用链一致性，不等同于独立精度标杆验证。
另一本地 ctypes ABI 检查为 9 通过、1 跳过、1 个因缺少 torch 报错。

环境：torch 2.7.1+cpu、torch_npu 2.7.1.post5、CANN 9.1，
Ascend950PR_9579 物理卡 2；使用独立安装目录，未替换共享环境。

证据保存在 A5：
`/home/Ensley/pr597-stable-20260918-pctOTJ/`，
包含源码清单、wheel、构建日志、两条后端日志与验证报告。
wheel SHA256：
`f281bb595be54d71b7c28c4d235f1a1548045cd712269e800e3d54e8752b9b52`。

性能采样见 [性能记录](../../chunk_kda_bwd_finalize/PERFORMANCE.md)。

## 历史精度结果与限制

历史 nan_fix_fused_scale 版本的 GPU FP64 双标杆结果为 600/600，
另通过 120 个压力用例与 18 个扩展用例；这些不是当前提交的全量复测。
原路径对照曾为 594/600 与 522/600，不应记为通过。

历史报告位于 246 服务器个人容器 admin123-gdn：
`/workspace/kda_l2_20260916/nan_fix_fused_scale/gpu_atk_accuracy_saved/atk_output/accuracy_2026-09-17-01-24-36-670473/report/accuracy_reports_2026-09-16-17-24-37.xlsx`。

重计算尾块存在重复运行稳定性问题，当前限制序列长度为 64 的倍数。
当前提交的全量双标杆、模型级、多流及 A2/A3 验证尚未完成；
本次结果不替代这些检查。

# FwdO NT-first 开发验证（2026-09-21）

基线：PR #701 的 `6ed1160c`。本批生产实现仅涉及 FwdO 的三处读取路径、
tiling/ACLNN 形状校验，以及 GDN 组合前向的调用适配；未改公式、dtype、
分核、缓冲、同步、dh 或融合 kernel 内部副本。

## 已确认结果

| 检查 | 结果 |
| --- | --- |
| Python 语法、git diff --check | 通过 |
| GDN 前向 aclnn_abi_contract.py | 通过，29 参数公开签名保持 |
| A2 两算子 run 包构建、独立 venv 安装 | 通过 |
| 加载 libcust_opapi 与本轮构建 MD5 | 一致：020be077cdc6df44844ec10a1655faa3 |
| 打包与安装 kernel 文件 SHA256 | 19/19 一致；补查了标准脚本跳过的 kernel 目录 |
| FwdO Stable ABI / ctypes parity | BNSD、NTD 两组通过，逐位一致 |
| GDN 原有前向 parity | 4 组通过；不代表 A5 组合路径已验证 |
| ATK CPU 标杆布局等价 | 3 个 shape × FP32/FP64 两路，共 6/6 输出逐元素一致 |

A2 环境为私有任务 venv、Ascend910B3、CANN 9.1.0；测试设备 4。
构建命令：`bash build.sh --soc=ascend910b --pkg --vendor_name=fla_npu --ops=chunk_fwd_o,chunk_gated_delta_rule_fwd -j12`。
原独立 FwdH 的测试/示例仅在 FwdO 输入处显式换布局；ATK 输入保持原随机值，
先按原 shape 生成再换轴，CPU 标杆只修改 h 索引。冻结 JSON/YAML 无 h shape，未改。

## A5 续跑结果（2026-09-21）

环境：`a5-ensley` / `sz-blue-950pr-13-241` / `Ensley`，Ascend950PR，
CANN 9.1.0；先激活 `wys_gdn`，再使用任务目录的独立 venv。
远端任务目录 `/home/Ensley/fwd-o-nt-first-20260921`。
原测试使用物理卡 0，恢复后的 ATK 和内存检查使用当时无进程的物理卡 4。
本地源码与构建时 `a5-source-manifest.json` 的 20 个文件哈希全部一致（更新本报告之前核对）。

| 检查 | 结果 |
| --- | --- |
| 两算子构建、安装及加载库核对 | 通过；libcust_opapi MD5 为 `651645d1cb8987b6eed57629ab1e862c` |
| 打包与安装 kernel 文件 | 19/19 一致 |
| NT-first 专项 Stable ABI | 11 passed，6 skipped |
| NT-first 专项 ctypes | 11 passed，6 skipped |
| 原有 CT 泛化检查 | 3/3 通过；为 CPU FP64 / NPU reference 双标杆，不是 GPU dump 双标杆 |
| 公开后端一致性 | 9 组逐位一致，4 组双方拒绝的配置显式跳过 |
| 新旧同输入比较 | 34/34 逐位一致：22 组 FwdO、12 组 GDN 组合前向 |
| GDN 组合前向 CPU reference | 12 组 o/ht 均通过原脚本阈值 rtol=0.02、atol=0.002 |
| ATK CPU 标杆布局等价 | 6/6 通过 |
| ATK 默认精度用例 | BF16 case 0 通过；FP16 case 1 被 A5 优化路径拒绝，整批未通过 |
| FwdO 目标 kernel memcheck | 3/3 通过，工具明确记录目标 ChunkFwdO kernel，均 No error detected |

专项的 6 项跳过来自 rank-3 输出要求 B=1，以及 A5 优化域要求 V=128、chunk=64。
同输入比较覆盖非零 gate、HV=NT/HV≠NT、多 batch、packed varlen、尾块、
兼容路径 V=256/chunk=128、A5 state_v_first/use_exp2，GDN 使用
Prepare → 共享 FwdH → 独立 FwdO 路径。内存检查为 3 组 BF16 dense
优化路径代表用例，不代表全部 TilingKey 或全部组合链路的内存验收。

恢复时发现原 `atk.log` 虽退出码为 0，实际加载了基础 conda 中的旧包并报告
`symbol not found: aclnnChunkFwdOGetWorkspaceSize`。恢复脚本改用任务 venv 的
`python -m atk`，保留 CANN PYTHONPATH，并优先指定任务 site-packages。
恢复后的默认 ATK case 0 精度为 True；case 1 的 FP16 与既有 A5 BSND 优化路径
仅 BF16 的约束冲突，报 161001，随后 ATK 汇总报 NoneType 错误。
不能以退出码 0 或 all.done 判定该批通过。未为此修改算子支持范围或冻结用例。

命令和证据：任务目录的 `run-a5.sh`、`extra-a5.sh`、`compare-a5.py`、
`recover-a5.sh`；结果为 `current-comparison.json`、`logs/` 及 `source/atk_output/`。
本地已保存 `fwd-o-nt-first-validation-20260921/a5-recovery-evidence.tar.gz`，包含
原始失败日志、恢复日志、34 组比较清单、3 组内存检查日志和 ATK 报告。
本次只完成 A5 定向验证，未声称完成全平台或正式合入验收。

## 尚未完成（A2 原有状态与验收限制）

- 新增 `tests/stable_abi/test_fwd_o_nt_first.py` 覆盖 dense/packed、HV=NT/HV≠NT、
  多 batch、尾块、V=256/chunk=128、A5 state_v_first、重复执行及旧布局拒绝。
- 首轮两个后端各有 1 个旧布局拒绝用例通过；其余 16 项因任务 venv 缺少 CT 而
  在 fixture 阶段报错，不能算精度通过。已安装已有 CT Tool 0.9.1 wheel。
- 临时重跑脚本的 CRLF 已在本地修正；随后 A2 SSH 连续超时，最后一次上传/启动
  是否完成以及专项、generalization 的最终结果需要重连确认。不得依赖旧的 tests.done。
- A5 已重连并完成上述定向验证；默认 ATK 整批仍受 FP16 配置问题影响。
- 现有 ATK FwdO executor 固定 use_exp2=True/BSND，为 A5 配置；本轮 A2 parity
  不能替代它的正式 ATK 验收。完整用例包、全 TilingKey 内存检查、GPU dump 双标杆、
  A3 与当前提交 CI 均未完成。

专项命令（加载本轮匹配 OPP，安装 CT Tool 后）：

```sh
python -m pytest -v tests/stable_abi/test_fwd_o_nt_first.py
FLA_NPU_STABLE_ABI=ctypes python -m pytest -v tests/stable_abi/test_fwd_o_nt_first.py
python torch_custom/fla_npu/test/test_npu_fwd_o_generalization.py
```

远端任务目录 `/data/wys/fwd-o-nt-first-20260921`，日志在 `logs/`，首轮记录保留在
`logs-first-attempt/`。重连后先查 run-a2.sh 进程、test-driver-final.log 和日志时间，
确认没有仍在执行的任务后再重跑。当前仅为开发中改动，不作为全平台通过或可合入结论。

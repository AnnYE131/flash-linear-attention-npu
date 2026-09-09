# chunk_gated_delta_rule_fwd ATK 验证

本目录验证公开接口 `fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd`。融合算子在 A2/A3
使用私有 `arch22` 实现，在 A5 使用私有 `arch35` 实现；A3 当前只具备注册和编译验收环境。

## 支持范围

- `q/k`：`[B,Hk,T,128]`，BF16/FP16。
- `v`：`[B,Hv,T,V]`，BF16/FP16，`V=128/256`，`Hv % Hk == 0`。
- `g/beta`：`[B,T,Hv]`；`g` 为 FP32，`beta` 与 q/k/v 同 dtype。
- `chunk_size`：64 或 128。
- 支持定长、变长、GVA、可选初始状态和可选最终状态。
- SoC：A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`)。

## 输出布局对齐

融合 DUT 的公开 `o` 为 `[B,T,Hv,V]`（BSND）；CPU golden 与六 ACLNN 标杆的原始
`o` 为 `[B,Hv,T,V]`（BNSD）。执行器在 `with_output=True` 且结果已回传 CPU 后，
按明确的角色把两路标杆的 `o` 转为 BSND；DUT 不再转置。不会按 shape 猜测布局，
因此 `T=Hv` 时也保留正确的轴映射。`A`、`g_cumsum` 和 state 的布局不变。

此适配不改变输入、CPU 数学公式、比较阈值或六算子计算，不向 NPU 性能路径添加转置。
结果附带 `o_source_layout` 与 `o_comparison_layout`，便于核对保存数据。

本地布局合同检查（需要 NumPy，无需 NPU、ATK 或 torch）：

```bash
python3 tests/atk/chunk_gated_delta_rule_fwd/scripts/test_output_contract.py
```

## 训练与推理输出模式

执行器读取可选 `disable_recompute` 属性，缺省为 `False`，保持原始 500 条用例行为。
`False` 比较 `o`、请求的 `final_state`、`g_cumsum` 和 `A`；`True` 只比较 `o` 与请求的
`final_state`，同时强制检查融合 DUT 公开返回四元组中的 `g_cumsum/A` 均为 `None`。
两路标杆仍完整计算原有数学结果，只选择当前模式需要的输出。省略辅助输出不代表内部跳过计算。

原始矩阵不改写。使用以下入口在已有运行目录派生模式和固定种子，case id、shape、输入值域、
属性及阈值保持不变；`--out` 拒绝覆盖已有文件：

```bash
python3 tests/atk/chunk_gated_delta_rule_fwd/scripts/prepare_output_mode_cases.py \
  --mode inference --seed 20260909 --out /absolute/run/inference_seed20260909.json
GDN_ATK_CASE_JSON=/absolute/run/inference_seed20260909.json \
  bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh 0
```

双模式验收分别使用 `training/inference`，各执行 `20260909/20260910/20260911` 三个固定种子。
每组 500 条，合计 3000 个 `(case, mode, seed)` 组合；必须记录实际完成数，不能将本地合同
检查视为 NPU 精度通过。模式切换不改变共同输出公式或值域；基线仍须通过原双标杆后才能迁移。

```bash
python3 tests/atk/chunk_gated_delta_rule_fwd/scripts/test_output_modes.py
```

## 精度标杆实现

精度使用 ATK 原生 `cv_fused_double_benchmark`：

1. NPU DUT：`chunk_gated_delta_rule_fwd`；
2. NPU benchmark：公开算子链 `chunk_local_cumsum`、`chunk_scaled_dot_kkt`、`solve_tri`、
   `recompute_w_u_fwd`、`chunk_gated_delta_rule_fwd_h`、`chunk_fwd_o`；
3. CPU golden：相同冻结输入上的 FP64 recurrence。

公开 `solve_tri` 来自 main 已合入的 PR 398；融合 kernel 不引用该公开实现，只有双标杆链路调用它。
比较阈值为最大相对误差比例 5、平均相对误差比例 1.5、均方根误差比例 1.5。

## 用例

- `atk_chunk_gated_delta_rule_fwd.json`：既有泛化 500 条冻结矩阵，五种场景各 100 条，
  覆盖 BF16/FP16、MHA/GVA、V128/V256、chunk 64/128、定长/变长及状态组合。
- `scripts/cases/legacy500_adapted.json`：既有 BF16/MHA 历史 500 条回归矩阵，不作为默认入口。
- `atk_chunk_gated_delta_rule_fwd_perf.json`：A5 两条模型 case：
  - 推理：`B=1,Hk=16,Hv=32,T=11274,K=V=128,chunk=64`，变长并输出最终状态；
  - 训练：`B=2,Hk=Hv=32,T=8192,K=V=128,chunk=64`，定长无状态输出。
- `atk_chunk_gated_delta_rule_fwd_mss.json`：从冻结矩阵抽取 6 条精简用例，覆盖 V128/V256、
  chunk 64/128、定长/变长、FP16/BF16 和状态输出。

A5 模型 shape 分别来源于 `推理model.csv` 和 `训练model.csv`，原文件 SHA256 为
`a8f21a5ddc23b824b2b5ccc625d33db95dd9441b33f2b0c0e3e313e72aeaa363` 与
`87e9bb1027c44eaf8cc2f5fc4d24256b22e0ff00c16162b5fb11aaaf24aca8ca`。

## TilingKey 覆盖

| TilingKey | 选择条件 | 普通/边界用例 | SoC | 实际选择证据 |
| --- | --- | --- | --- | --- |
| 1 | `V=128` | MSS 0、2、4 | A2/A3/A5 | 本 PR 硬件门禁补录 |
| 2 | `V=256` | MSS 1、3、5 | A2/A3/A5 | 本 PR 硬件门禁补录 |

## 执行

先执行不依赖 NPU/ATK 的 ACLNN ABI 合同，确认公开参数顺序、ctypes 类型和默认路径映射：

```bash
python3 tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py
```

公开 `aclnnChunkGatedDeltaRuleFwd` 保留完整扩展 ABI。当前 Phase6 默认路径使用
`layout=BNSD`、`useExp2=false`、`allowNegEigval=false`、`stateVFirst=false`，且
`aLog/dtBias` 与扩展中间输出为空；`finalStateOutOptional` 是否为空决定是否输出 final state。
尚未实现的扩展组合会显式返回参数错误，不会静默忽略。

正式 500 条双标杆精度使用可恢复分片入口；默认每 25 条启动一个 fresh ATK 进程，避免
六算子 benchmark 长进程状态累积：

```bash
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh 0
```

冒烟或单分片可直接使用三节点入口（默认 `-mt 5`）：

```bash
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_double_benchmark.sh 0
```

复跑历史矩阵：

```bash
GDN_ATK_CASE_JSON="$PWD/tests/atk/chunk_gated_delta_rule_fwd/scripts/cases/legacy500_adapted.json" \
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh 0
```

性能、确定性和内存检测仍使用仓内统一入口：

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=0 -scope=mssanitizer
```

正式结论必须记录代码 commit、ATK/CANN 版本、SoC、实际加载的 OPP、case JSON 哈希和原始报告。

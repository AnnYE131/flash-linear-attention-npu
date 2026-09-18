# Development validation

## Stable-ABI integration (2026-09-18)

This integration merges upstream main `5ceca6f2` (including Stable-ABI #559).
Only the KDA backward host adapters and shared Python policy are changed;
the V2 ACLNN declaration and device implementations remain unchanged.

Local offline validation:

- `stable_coverage.py`, `op_abi_parity.py`, `op_api_parity.py` and
  `stable_ctypes_fallbacks.py`: passed; no added fallback or public op.
- `python -m unittest tests.test_stable_gates`:
  29 tests run, 28 passed, 1 skipped by the existing gate suite.
- V2 source header vs ctypes table and Stable-ABI call: matched.
- `git diff --check`: passed.

The additional existing `test_aclnn_ctypes_abi.py` suite ran 11 tests:
9 passed, 1 skipped, and the Dhu test could not run because local Python has no
`torch` installation (`ModuleNotFoundError`). This is not recorded as a suite pass.

`scenario_chunk_kda_bwd_optimized` is registered in both Stable-ABI regression
drivers and the `kda` group. Its seven host-parity cases cover dense, single
token, dense tail, packed, packed empty sequence/tail, dense recompute and
packed recompute, including beta dtypes, rstd and absent/flat/matrix dt_bias.
These synthetic saved-cache inputs test host parity, not end-to-end numerical
accuracy against a mathematical reference.

Device verification completed on the private A5 account in `wys_gdn`:

- Host `sz-blue-950pr-13-241`, device 2 (`Ascend950PR_9579`), CANN 9.1.0,
  torch 2.7.1+cpu and torch_npu 2.7.1.post5.
- Built with `FLA_NPU_SOC=ascend950 FLA_NPU_OPS=chunk_kda_bwd` and the default
  Stable-ABI backend. The launcher and complete wheel both built successfully.
- Isolated `pip --target` installation exports `aclnnChunkKdaBwd` and
  `aclnnChunkKdaBwdV2`, plus both GetWorkspaceSize symbols. Prepare, Finalize,
  Recompute, FwdH and Dhu kernels are included. The removed Python module is absent.
- The same 29 offline tests passed on A5 with no skips.
- Public `chunk_kda_bwd` calls: **8/8 passed per backend**, covering legacy dense
  plus the seven optimized cases above. Stable vs ctypes outputs are bitwise
  identical, with expected shapes, dtypes and optional slots. All optimized
  outputs are finite. Two unsupported combinations are rejected by both sides.
- Independent processes selecting `FLA_NPU_STABLE_ABI=stable` and `ctypes`
  reproduce bitwise-identical outputs across all eight calls. `BACKENDS` records
  the requested backend and `FALLBACKS` is empty.
- Only these measured passes were added to the `Ascend950PR_9579` scenario
  baseline; all pre-existing device records are retained. This was a targeted
  KDA backward run, not the full repository regression or the 600-case ATK suite.

Evidence directory on A5: `/home/Ensley/pr597-stable-20260918-pctOTJ`.
`build.log`, `offline.log`, `stable.log`, `ctypes.log`, `validation.log` and
both `*_report.json` files record the checks. `pr597_validate.py` and
`run_validation.sh` reproduce the run. `pr597-stable-a5-source.tar.manifest.json`
records SHA256 for all 1597 transferred source files before building.

Wheel: `source/dist/flash_linear_attention_npu-26.7.0.dev0-950.x86_64-py3-none-linux_x86_64.whl`.
Final wheel SHA256 (after runtime metadata pinning):
`f281bb595be54d71b7c28c4d235f1a1548045cd712269e800e3d54e8752b9b52`.
Installed opapi SHA256:
`a4796ae714e586c3bfb77fd5fc96e2860d1fa5b6b0a7349597b20c62086beff8`.

These results establish build, packaging and host-call parity for the tested
matrix. They do not replace independent numerical qualification, full-model
training tests, performance measurements or A2/A3 validation.

### Targeted msprof timing

The same installed wheel was profiled on physical A5 device 2 with
`B=1, H=96, T=8192, K=V=128, chunk_size=64, disable_recompute=True`,
`implementation="optimized"` and the Stable-ABI backend. Both runs used identical
synthetic saved caches, BF16 token tensors/beta, FP32 gates/parameters and
`lower_bound=-1.0`. Only the presence of the FP32 q/k rstd pair changed.
Each run excluded 10 warmup calls and measured the next 20 calls.

`msprof --task-time=on --ai-core=off --runtime-api=off --ascendcl=off` captured
the following `Task Duration(us)` values:

| Kernel | No norm mean | Norm mean | No norm median | Norm median |
| --- | ---: | ---: | ---: | ---: |
| ChunkKdaBwdPrepare | 1111.544 | 1109.413 | 1110.983 | 1108.890 |
| ChunkGatedDeltaRuleBwdDhu | 1467.921 | 1487.994 | 1446.978 | 1488.054 |
| ChunkKdaBwdFinalize | 7024.338 | 7603.624 | 6769.528 | 7579.844 |
| Per-call sum of three kernels | 9603.804 | 10201.030 | 9323.068 | 10184.221 |

L2Norm backward remains fused into Finalize: both runs contain exactly three
kernels per call. The measured mean sum increases by 0.597 ms (6.2%). Samples
fluctuate: the no-norm sums range from 9.269 to 10.493 ms, and norm sums from
9.629 to 10.913 ms. These timing-only runs do not normalize frequency across
processes and do not include Python overhead or Task Wait Time in the table.
They are not a full-model latency measurement or a new numerical qualification.

Raw profiles, filtered op summaries, the benchmark script and aggregation script
are archived under `msprof-h96t8192/` in the A5 evidence directory above.

## Current saved-chain qualification (2026-09-17)

The historical results below describe the initial integration, before the NaN
repair. They are not the status of the current candidate.

The repaired `disable_recompute=True` chain uses stable 32-row exponent bands,
FP32 prepare output, direct NZ operands and synchronized per-band postprocessing.
The `nan_fix_causal_rows` isolated wheel passed **600/600 ATK dual-reference cases**:
200 frozen profiles at three seeds, with actual FLA GPU outputs and an independent
same-cache FP64 golden. The ratio thresholds remain 5/1.5/1.5. All GPU inputs are
hash-checked against the exact DUT caches; regenerated cross-host caches that
differed were rejected and replaced with the exact inputs before GPU execution.
The GPU runs on `gpu-server` (RTX 3060 Ti), FLA commit
`51a8c0a88b9e0e7a4409e3670db771fd58c389c7`. All 370 loaded FLA source files match
that commit. ATK version on the test machine is 26.7.8.

Report under `/workspace/kda_l2_20260916` in `server-246/admin123-gdn`:
`nan_fix_causal_rows/gpu_atk_accuracy_saved/atk_output/accuracy_2026-09-17-00-20-05-559142/report/accuracy_reports_2026-09-16-16-20-06.xlsx`.
A deliberately zeroed DUT fails the GPU dual-reference sentinel, as required.
This is a ctypes L2 execution test, not a pyaclnn ABI or full-model qualification.

Subsequent `nan_fix_compact_ws`, `nan_fix_stream_outputs`, and
`nan_fix_stream_intra` wheels each have 600/600 bitwise output-equivalence checks
against the passing candidate. `stream_intra` also passes 120 extreme-gate tests
and 18 optional-input/packed-empty-sequence cases repeated three times.

The earlier legacy-NPU-control reports remain **594/600** (recurrence golden)
and **522/600** (same-cache golden). They have not become passes; tiny parameter
gradient cancellation and the different intermediate precision are why those
reports must be kept separate from the actual GPU control results.

The selected wheel is `nan_fix_fused_scale`, SHA256
`9ebbf3f0532121e77e1a9bf1d17b64265d1cd06baa9be555beb88c232ca6cc04`.
It passes 600-case bitwise equivalence to the GPU-ATK-qualified candidate,
600/600 same-cache FP64 engineering tests, 120/120 stress cases and 18/18
extended cases with three identical repeats. Direct actual-GPU ATK replay on
this selected wheel passes **600/600**, with zero execution failures and
unchanged thresholds. Report:
`nan_fix_fused_scale/gpu_atk_accuracy_saved/atk_output/accuracy_2026-09-17-01-24-36-670473/report/accuracy_reports_2026-09-16-17-24-37.xlsx`.
All three full-size launches are finite; the fixed heads [0,H/2,H-1] pass
the same-cache FP64 checks while the DUT retains its original H/T tiling.

The frozen device-time limits are
3.33585, 9.70410, and 20.31855 ms for H64/T4096, H96/T8192, and H96/T16384,
respectively (B1, K=V128). The selected wheel passes all three independent
trials of each shape. The three trial ranges are 3.250883–3.254471,
9.328042–9.468854, and 19.802146–20.134261 ms. Even the slowest trial is within
the 5% limit (+2.44%, +2.45%, +4.05%, respectively).
Use the sum of the three per-kernel medians after 10 warmups and 20 measured
calls; do not substitute the best sample or Python wall time.

Concurrent-stream behavior, full GDN regression and model regression are not yet
qualified. Recompute/fwd_h/dhu and the legacy implementation remain unchanged.

## Historical integration evidence (pre-NaN repair)

2026-09-16, branch feat/kda-bwd-optimized-l2.

Sources: integration base 9c0ffc175da6f924492282e7c07046639ddfc267;
prepare from prepare-conflict-fix 9eed6ee862c80162983d89ac14890a8b99472c18;
finalize from 24ebeee316f5bd131d3b39cf68befcf4d7fe867a plus the separately
validated beta/dbeta dtype extension. Existing 1200-case finalize evidence does
not establish whole-chain precision.

Development runs in server-246's personal admin123-gdn container, Conda wys_gdn,
under /workspace/kda_l2_20260916. No global package replacement is required.

- Python syntax checks pass.
- Five Host-only routing/metadata tests pass.
- Initial build produced a wheel but lacked V2 exports because the source-level
  compile definition was attached to the wrong CMake directory. Corrected to
  the shared opapi object target's directory.
- Subsequent C++ compilation exposed helper names conflicting with op::Shape
  and op::Tensor; renamed to MakeShape and MatchesTensor.
- Final Ascend950 wheel builds successfully. SHA256:
  `c983b8f3354198afb36232fc63120dc7e81246ff82d1cfc70e5c2c5ff1858979`.
- Saved-mode composition: 24 cases pass across three seeds, dense 64/65 and
  packed [1,63,64,65]/[65,0,1,63], both beta types and optional norm/bias.
  Outputs match separately launched prepare/dhu/finalize exactly.
- Aligned recompute composition: 18 cases pass across three seeds, dense 64/128
  and packed [64,0,128]. Recomputed and saved-mode outputs match exactly.
- Normalization cases also cover BF16 raw_g/A_log promotion. Dense 64 non-norm
  cases compare default legacy against explicit legacy, including legacy dtypes.
- Independent whole-chain GPU/ATK precision, concurrency, model regression and
  end-to-end performance remain unvalidated. These composition checks are not
  an independent mathematical oracle or the requested 200-case precision suite.

In the unrestricted mixed regression, consecutive existing recompute calls on
a 65-token case produced different w (maximum absolute difference 0.0680541992),
before that case invoked the new L2. A standalone 65-token diagnostic with
100 repetitions per seed over three seeds did not reproduce it. The root cause
is not established: preserve the mixed-workload context when reproducing.
No failure tensor was captured in the positive reproduction; the log and exact
test source are retained. Per user instruction, recompute is not repaired here.
The V2 wrapper and C entry conservatively reject non-64-aligned recompute
lengths. Saved mode still accepts tails. This guard does not prove all upstream
recompute defects are confined to tails.

Evidence on server-246, admin123-gdn: /workspace/kda_l2_20260916 contains
saved_final.json, aligned_final.json, unrestricted_replay.log,
unrestricted_test_composition.py and unrestricted_site.tar.gz (the pre-guard
environment). The current site is the final guarded build. To replay the old
failure, extract the archived environment separately and point PYTHONPATH and
ASCEND_CUSTOM_OPP_PATH to it; do not overwrite the final site. The simplified
negative diagnostic is tests/integration/reproduce_recompute_tail.py.

Follow-up saved-chain independent-reference testing on 2026-09-16 failed:
finite CPU forward saves and finite prepare/dhu intermediates can produce NaN
in finalize. At B1/H4/T63, gk reaches about -241, exp2(gk) underflows and Stage5's
reciprocal becomes nonfinite. Reproduced at three seeds; the legacy entry stays
finite on those inputs. The prior composition checks compared identical L0s
and did not expose this value-range limitation. See tests/atk/chunk_kda_bwd_optimized
for the fixed matrix, reference provenance and failure status. The planned
600-run matrix is incomplete; no full precision pass is claimed.

The requested B1/H64/T4096, H96/T8192 and H96/T16384 ATK Device timing comparison
measured legacy/V2 at 5.9751/3.1766, 17.5693/9.2419 and 36.3812/19.3506 ms.
These are pre-fix measurements, not a correctness-qualified acceleration result.

Do not describe this branch as delivery/merge ready until the remaining checks
are complete. Production code in recompute/fwd_h/dhu is unchanged.

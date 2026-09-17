## Saved-chain numerical stability (2026-09-16)

Real forward gk is cumulative within each chunk. At C=64 and safe-gate
lower_bound=-5, its magnitude can approach 462 in log2 units. The previous
k/exp2(gk) factorization overflows on valid real caches; small random standalone
gk tests do not establish correctness for this interface.

The selected candidate retains the two-head owner schedule and existing BF16
high/residual GEMMs. Intra GEMMs operate on 32 output rows per band. For each
feature, center=(first(g_band)+last(g_band))/2. The supported safe-gate cumsum
is nonincreasing, so its endpoints equal the band maximum/minimum exactly.
Lower-side k uses exp2(center-g),
upper-side q and beta*k use exp2(g-center). Inputs outside the corresponding
causal support are zeroed before exponentiation. Band results restore the
opposite factor. Complete chunks have the same mathematical GEMM FLOPs but
more result handshakes than the previous single-band implementation.

A=tril(dAqk,-1) and D=-tril(dAkk_raw,-1) are scaled by exactly 2^16 before the
existing BF16 high/residual decomposition. Results are scaled by 2^-16 after
restoring the band gate. This protects small residual products against
underflow in saturated gates. A/D are published to L1 once per head and reused
across bands. This is not the rejected span=192 direct-Vector fallback.
No gate clamp, causal-term truncation, or additional kernel is introduced.

The diagonal dAqk[i,i] is retained separately: dq adds diagonal*k, dk adds
diagonal*q. Its gate derivative is exactly zero, so it does not enter the
subtraction q*dq-k*dk. Off-diagonal local gradients are:

```
dk_local = beta*left + right
db_local = sum_K(k*left)
dg_local = q*dq_offdiag + k*(beta*left-right)
```

Prepare now preserves FP32 Cube dAqk through mask/scale and GM output, removing
the BF16 round trip without adding MMADs. Its UB allocation grows from 112 to
128 KiB. Finalize remains at 248 KiB UB and 128 KiB L1 per owner; diagonal data
occupies [210,210.25) KiB UB through the last band. Workspace uses four 160-KiB
slots per AIC (two heads, two alternating windows), rather than eight 224-KiB
slots. Host/device strides agree, including the parameter-partial base offset.
The former dq/left/right GM intermediates are no longer needed.

Final outputs bypass L2. The once-read raw gate, dAqk and dqRaw also bypass L2;
q/k/beta are loaded once and retained in UB. On Cube, Akk/vNew/dvScan bypass
L2 and remain in L1 through their consumers. Shared state, gate and workspace
traffic keeps the default cache policy. This changes data placement only,
without changing arithmetic, output rounding or launch count.
Prepare also bypasses L2 for its once-read inputs and full-size outputs.

All five operand pairs are produced directly in NZ order and copied to L1
contiguously. Noncausal rows publish zero planes directly, without redundant
multiply/cast/residual work. dq and left share kNeg and are stacked in one
Cube result; the two right products are concatenated along K.

Each band publishes L1 with MTE3 before LOCAL_READY. After DQ_READY, AIV rescales
dq/left, completes Stage7, releases KE_READY, then consumes right and completes
Stage9/10. dq/dk/db go directly to their final outputs. Only the accumulated dg
band is written to workspace. Stage11 reloads the complete dg chunk and applies
the unchanged reverse scan and gate derivative. Stage12 reduces parameter
partials as before. No extra kernel is launched.
Stage7/9 restore the band factors directly in registers, retaining Mul/Div then
2^-16 scaling before the original gradient arithmetic. This removes the separate
scaling passes through UB without changing the 600-case output hashes.

Band postprocessing keeps q/k intact for the next Stage5. Its temporary UB
regions are dg [112,128), dq base [128,144), dk base [216,232), dk output
[232,240), and dq output [240,248) KiB. Scalar scratch uses 211–215 KiB.
These regions are disjoint from either Cube result slot after Stage5's L1
publication drains. Consecutive Vector consumers use VEC_STORE→VEC_LOAD
barriers. MTE3_MTE2 protects both next-band reuse and the final Stage11 reload;
an MTE3_V wait alone does not order the later MTE2 writes.

The V2 public entry now uses standard L2_DFX_PHASE_1/2 initialization. This fixes
cold BF16-to-FP32 Cast tiling coreNum=0 on CANN 9.1; moving Cast alone did not.
Saved-mode casts remain located at their finalize consumer. The API does not
require caller warmup. Recompute, fwd_h, dhu and the legacy branch are unchanged.

Two independent FP64 references are reported separately: same-cache derivatives
for interface accuracy, and full recurrence for mathematical end-to-end error
including forward-cache rounding. Legacy NPU and actual FLA GPU controls are
reported separately, with public output dtypes aligned. GPU replay checks the
complete input/cache SHA256 before comparison; all reconstructed GPU outputs
also match their original GPU-side hashes. The legacy-control failures remain
recorded and are not relabeled as GPU-control results.
See optimized_validation.md for measured counts, scope and remaining limits.

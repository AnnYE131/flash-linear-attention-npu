# Optimized KDA backward integration

The existing Python entry accepts three additional keyword-only arguments:
`implementation="auto"`, `q_rstd=None`, `k_rstd=None`. Existing calls retain the
legacy ABI, output types and fallback behavior. `implementation="optimized"`
selects `aclnnChunkKdaBwdV2`; auto selects V2 for a complete rstd pair or
`disable_recompute=False`. Explicit legacy rejects rstd. Unsupported optimized
requests fail before launch and never fall back after a runtime error.

Both the default Stable-ABI backend and the ctypes reference expose this same
Python signature and selection policy. The Stable-ABI wrapper dispatches through
the existing `npu_chunk_kda_bwd` registration; its C++ adapter selects the legacy
ACLNN symbol or `aclnnChunkKdaBwdV2`. There is no additional public Python op.
The internal dispatcher schema adds an implementation flag and optional rstd
slots, and permits absent gk in recompute mode. Rebuild the Stable-ABI library
together with the Python package when upgrading this branch.

Shared input validation and metadata normalization live in the existing
`_kda_policy.py`. `_stable.py` and `_aclnn_ctypes.py` each implement their own
launch. The separate `_kda_bwd_optimized.py` module is no longer needed.

```python
from fla_npu.ops.ascendc import chunk_kda_bwd

dq, dk, dv, db, dg, dh0, dA, dbias = chunk_kda_bwd(
    q, k, v, beta, gk, Aqk, Akk, w, qg, kg, v_new, h, d_o, scale,
    raw_g=raw_g, A_log=A_log, dt_bias=dt_bias,
    cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    use_gate_in_kernel=True, safe_gate=True, use_exp2=True,
    disable_recompute=True, implementation="optimized",
    q_rstd=q_rstd, k_rstd=k_rstd,
)
```

For recomputation set `disable_recompute=False` and pass None for
`gk,w,qg,kg,v_new,h`. Aqk/Akk remain required. The L2 executor allocates internal u.

| Contract | Optimized path |
|---|---|
| Platform | Ascend950/A5 |
| Dimensions | Hq=Hv, K=V=128, chunk_size=64, positive B/H/T |
| Recompute heads | H<=256, H divisible by 8, until A_log tail storage is validated |
| Recompute lengths | Every sequence length divisible by 64; tails temporarily rejected pending upstream repeatability repair |
| Token storage | contiguous dense [B,H,T,D], packed [H,T,D] |
| h storage | head-major [B,H,Nc,K,V] or [H,Nc,K,V] |
| q/k/v/dO, Aqk/Akk, token/state caches | BF16, except gk FP32 |
| beta / db | BF16 or FP32, matching dtype |
| dq/dk/dv | BF16 |
| raw_g / A_log | BF16 or FP32; cast to FP32 inside executor as needed |
| dt_bias | optional FP32 H*128 elements, presented to V2 as [H,128] |
| dg/dA/dbias | FP32; dbias None when dt_bias absent |
| q_rstd/k_rstd | optional pair, FP32 token prefix shape |
| Gate | safe_gate/use_gate_in_kernel/use_exp2 true; -5<=lower_bound<0 |
| State | initial_state/dht unsupported; dh0 None; state_v_first false |
| Metadata | Host int64 arrays in canonical sequence-major chunk order |

q/k are exactly the effective inputs saved by forward. If forward normalized
them, pass the normalized BF16 tensors and saved rstd. Finalize applies norm
backward, and the caller must not apply it again. Without rstd, gradients are
with respect to supplied q/k. No raw_q/raw_k reconstruction is performed.

The Python wrapper removes empty sequences from metadata and renumbers sequence
IDs without changing token/chunk storage. Direct V2 callers must supply this
nonempty canonical form. Entirely empty T=0 is rejected. No sequence state
mapping is supported. Device-resident metadata is rejected rather than silently
synchronizing it to the Host.

The eight public output slots remain `(dq,dk,dv,db,dg,dh0,dA,dbias)`.
The C V2 ABI retains legacy parameter order and inserts q/k rstd immediately
before output pointers; it is a distinct symbol and never changes the old ABI.
Its exact declaration is `op_host/op_api/aclnn_chunk_kda_bwd_v2.h`.
V2 materializes missing dt_bias and discarded parameter outputs internally.

Limitations and validation results are recorded in optimized_validation.md.

"""Dense head-first reverse chunk scans used by the GDN backward path.

The Ascend C implementation remains the reference for varlen inputs and for
all other cumsum modes.  This kernel only changes the dense ``[B, H, T]``
reverse scan, where one program can cover several independent chunks without
changing the FP32 reduction order inside a chunk.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dense_reverse_chunk_scan(
    x,
    out,
    T: tl.constexpr,
    H: tl.constexpr,
    GROUPS: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    # Flatten the logical (batch, head, chunk-group) grid.  A2's Triton
    # launcher can leave middle IDs unlaunched for a 3-D grid; decoding a
    # single linear ID keeps every head covered.
    linear_pid = tl.program_id(0)
    chunk_group = linear_pid % GROUPS
    head_linear = linear_pid // GROUPS
    head = head_linear % H
    batch = head_linear // H
    chunk_base = chunk_group * BC
    chunk_ids = chunk_base + tl.arange(0, BC)
    token = tl.arange(0, BT)
    time_offsets = chunk_ids[:, None] * BT + token[None, :]
    valid = time_offsets < T
    offsets = (batch * H + head) * T + time_offsets
    values = tl.load(x + offsets, mask=valid, other=0.0)
    result = tl.cumsum(values, axis=1, reverse=True)
    tl.store(out + offsets, result, mask=valid)


def reverse_scan_dense_head_first(
    x: torch.Tensor,
    chunk_size: int,
    chunks_per_program: int = 4,
) -> torch.Tensor:
    """Reverse cumulative sum independently inside each dense time chunk.

    ``x`` is contiguous FP32 ``[B, H, T]``.  ``chunks_per_program`` is a
    workload class knob shared by chunk64/128; it is deliberately independent
    of B, T, or the head counts so the dispatch does not encode a model shape.
    """

    if x.ndim != 3:
        raise ValueError(f"expected rank-3 [B, H, T], got {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise TypeError(f"expected float32 input, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("dense reverse scan requires contiguous input")
    if chunk_size not in (64, 128):
        raise ValueError(f"chunk_size must be 64 or 128, got {chunk_size}")
    if chunks_per_program not in (1, 2, 4, 8):
        raise ValueError(
            f"chunks_per_program must be one of 1/2/4/8, got {chunks_per_program}"
        )

    batch, heads, tokens = x.shape
    out = torch.empty_like(x)
    chunk_count = triton.cdiv(tokens, chunk_size)
    groups = triton.cdiv(chunk_count, chunks_per_program)
    grid = (batch * heads * groups,)
    _dense_reverse_chunk_scan[grid](
        x,
        out,
        T=tokens,
        H=heads,
        GROUPS=groups,
        BT=chunk_size,
        BC=chunks_per_program,
        num_warps=4,
    )
    return out

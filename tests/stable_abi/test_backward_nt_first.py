"""Dhu NT-first storage and internal Cube readback, using w=0 identities.

Run with the installed wheel under both Stable ABI and ctypes. Nonzero dht
is intentionally outside this layout regression: the existing kernel ignores it.
"""
import pytest
import torch

pytest.importorskip("torch_npu")
from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_bwd_dhu


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("svf", [False, True])
@pytest.mark.parametrize("heads", [2, 3])
@pytest.mark.parametrize("vdim", [128, 256])
def test_dhu_state_storage(packed, svf, heads, vdim):
    torch.manual_seed(701)
    b, t = (1, 130) if packed else (2, 129)
    cu = [0, 1, 65, 130] if packed else None
    ci = [0, 0, 1, 0, 2, 0, 2, 1] if packed else None
    q = (torch.randn(b, 1, t, 128) * .1).bfloat16()
    k = (torch.randn_like(q.float()) * .1).bfloat16()
    do = (torch.randn(b, heads, t, vdim) * .1).bfloat16()
    dv = torch.zeros_like(do)
    tail = (vdim, 128) if svf else (128, vdim)
    initial = torch.zeros(3 if packed else b, heads, *tail, dtype=q.dtype)
    actual = npu_chunk_gated_delta_rule_bwd_dhu(
        q.npu(), k.npu(), torch.zeros(b, heads, t, 128, dtype=q.dtype, device="npu"),
        do.npu(), dv.npu(), 1., 64, g=torch.zeros(b, heads, t, device="npu"),
        h0=initial.npu(), cu_seqlens=cu, chunk_indices=ci,
        transpose_state_layout=svf)
    expected = torch.zeros((4, heads, 128, vdim) if packed else (b, 3, heads, 128, vdim))
    expected0 = torch.zeros(3 if packed else b, heads, 128, vdim)
    expected_dv = torch.zeros_like(do.float())
    for batch in range(b):
        intervals = list(zip(cu, cu[1:])) if packed else [(0, t)]
        chunk_base = 0
        for seq, (left, right) in enumerate(intervals):
            state = torch.zeros(heads, 128, vdim, dtype=torch.float64)
            starts = list(range(left, right, 64))
            for chunk in reversed(range(len(starts))):
                start, end = starts[chunk], min(starts[chunk] + 64, right)
                if packed:
                    expected[chunk_base + chunk] = state
                else:
                    expected[batch, chunk] = state
                expected_dv[batch, :, start:end] = k[batch, :, start:end].double() @ state
                state = state + q[batch, :, start:end].double().transpose(-1, -2) @ do[batch, :, start:end].double()
            expected0[seq if packed else batch] = state
            chunk_base += len(starts)
    if svf:
        expected, expected0 = expected.transpose(-1, -2), expected0.transpose(-1, -2)
    torch.npu.synchronize()
    for result, target in zip(actual, (expected, expected0, expected_dv)):
        assert result.shape == target.shape
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result.cpu().float(), target.bfloat16().float(), rtol=.03, atol=.006)

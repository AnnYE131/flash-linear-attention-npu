"""A5 public FwdH -> FwdO/Finalize, without head/chunk or batch adaptation."""
import pytest
import torch

pytest.importorskip("torch_npu")
from fla_npu.ops.ascendc import chunk_fwd_h, chunk_fwd_o, chunk_kda_fwd_finalize


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("svf", [False, True])
def test_shared_h_consumers(packed, svf):
    if "950" not in torch.npu.get_device_name(torch.npu.current_device()):
        pytest.skip("Both consumer paths with state_v_first require A5")
    torch.manual_seed(20260922)
    b, t, heads = (1, 130, 2) if packed else (2, 129, 3)
    cu = [0, 1, 65, 130] if packed else None
    ci = [0, 0, 1, 0, 2, 0, 2, 1] if packed else None
    q = (torch.randn(b, heads, t, 128) * .1).bfloat16()
    initial = (torch.randn(3 if packed else b, heads, 128, 128) * .1).bfloat16()
    initial_physical = initial.transpose(-1, -2).contiguous() if svf else initial
    zero = torch.zeros_like(q).npu()
    gate = torch.zeros(b, heads, t, device="npu")
    h, v_new, _ = chunk_fwd_h(zero, zero, zero, g=gate,
        initial_state=initial_physical.npu(), cu_seqlens=cu, chunk_indices=ci,
        state_v_first=svf)
    assert h.shape == ((4, heads, 128, 128) if packed else (b, 3, heads, 128, 128))
    expected = torch.empty(b, heads, t, 128, dtype=torch.float64)
    for batch in range(b):
        intervals = [(0, t)] if cu is None else list(zip(cu, cu[1:]))
        for seq, (left, right) in enumerate(intervals):
            for head in range(heads):
                state = initial[seq if packed else batch, head].double()
                expected[batch, head, left:right] = q[batch, head, left:right].double() @ state
    q_npu = q.npu()
    o = chunk_fwd_o(q_npu, zero, v_new, h, 1.0, g=gate,
        cu_seqlens=cu, chunk_indices=ci, chunk_size=64,
        transpose_state_layout=svf, output_layout="BSND")
    a = torch.zeros(b, heads, t, 64, dtype=q.dtype, device="npu")
    finalized = chunk_kda_fwd_finalize(q_npu, a, v_new, h,
        cu_seqlens=cu, chunk_indices=ci, state_v_first=svf, output_layout="BNSD")
    torch.npu.synchronize()
    for result in (o.transpose(1, 2), finalized):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result.cpu().float(), expected.bfloat16().float(),
                                   rtol=.02, atol=.002)

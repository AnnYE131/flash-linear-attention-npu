"""A5 前向状态直连与导出检查。"""
import pytest
import torch

pytest.importorskip("torch_npu")
from fla_npu.ops.ascendc import chunk_fwd_h, chunk_gated_delta_rule_fwd_h, chunk_fwd_o, chunk_kda_fwd_finalize, chunk_gated_delta_rule_fwd


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("svf", [False, True])
@pytest.mark.parametrize("producer", [chunk_fwd_h, chunk_gated_delta_rule_fwd_h])
def test_shared_h_consumers(packed, svf, producer):
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
    h, v_new, _ = producer(zero, zero, zero, g=gate,
        initial_state=initial_physical.npu(), cu_seqlens=cu, chunk_indices=ci,
        state_v_first=svf)
    assert h.shape == (b, 4 if packed else 3, heads, 128, 128)
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


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("state_v_first", [False, True])
@pytest.mark.parametrize("heads", [2, 3])  # 覆盖 NT 与 HV 相等和不等。
def test_exported_h_shape_and_content(packed, state_v_first, heads):
    if "950" not in torch.npu.get_device_name(torch.npu.current_device()):
        pytest.skip("GDN return_intermediate_states requires the A5 prepare path")
    torch.manual_seed(20260919)
    b, t = (1, 130) if packed else (2, 129)
    cu = [0, 1, 65, 130] if packed else None
    indices = [0, 0, 1, 0, 2, 0, 2, 1] if packed else None
    seq = 3 if packed else b
    q, k = [(torch.randn(b, 1, t, 128) * .01).bfloat16().npu() for _ in range(2)]
    v = (torch.randn(b, heads, t, 128) * .01).bfloat16().npu()
    # beta=g=0，h 保持为各序列初态。
    initial_kv = (torch.randn(seq, heads, 128, 128) * .01).bfloat16()
    initial = initial_kv.transpose(-1, -2).contiguous() if state_v_first else initial_kv
    outputs = chunk_gated_delta_rule_fwd(
        q, k, v, torch.zeros(b, t, heads, device="npu"),
        torch.zeros(b, t, heads, device="npu"),
        initial_state=initial.npu(), output_final_state=True,
        chunk_size=64, cu_seqlens=cu, chunk_indices=indices,
        use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False, disable_recompute=False,
        return_intermediate_states=True, state_v_first=state_v_first, layout="BNSD",
    )
    torch.npu.synchronize()
    h = outputs[5].cpu()
    expected = (torch.stack([initial[s] for s in (0, 1, 2, 2)]).unsqueeze(0) if packed
                else initial.unsqueeze(1).expand(b, 3, heads, 128, 128))
    assert h.shape == expected.shape
    assert h.is_contiguous()
    torch.testing.assert_close(h, expected, rtol=0, atol=0)
    torch.testing.assert_close(outputs[1].cpu(), initial, rtol=0, atol=0)

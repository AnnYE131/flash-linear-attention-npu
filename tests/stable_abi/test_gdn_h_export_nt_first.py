"""Target-contract regression for GDN h export (P2 must make this pass).

Run with the matching wheel/OPP on A5, then repeat with
FLA_NPU_STABLE_ABI=ctypes. No xfail: pending migration must remain visible.
"""
import pytest
import torch

pytest.importorskip("torch_npu")
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd


@pytest.mark.parametrize("seed", [20260919, 20260920, 20260921])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("state_v_first", [False, True])
@pytest.mark.parametrize("heads", [2, 3])  # dense NT=3; test HV!=NT and HV==NT.
def test_exported_h_shape_and_content(seed, packed, state_v_first, heads):
    if "950" not in torch.npu.get_device_name(torch.npu.current_device()):
        pytest.skip("GDN return_intermediate_states requires the A5 prepare path")
    torch.manual_seed(seed)
    b, t = (1, 130) if packed else (2, 129)
    cu = [0, 1, 65, 130] if packed else None
    indices = [0, 0, 1, 0, 2, 0, 2, 1] if packed else None
    seq = 3 if packed else b
    q, k = [(torch.randn(b, 1, t, 128) * .01).bfloat16().npu() for _ in range(2)]
    v = (torch.randn(b, heads, t, 128) * .01).bfloat16().npu()
    # beta=0, g=0 => w=u=0: h remains exactly the sequence's initial state.
    # Different sequence/head and nonsymmetric K/V values expose axis errors.
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
    expected = (torch.stack([initial[s] for s in (0, 1, 2, 2)]) if packed
                else initial.unsqueeze(1).expand(b, 3, heads, 128, 128))
    assert h.shape == expected.shape
    assert h.is_contiguous()
    torch.testing.assert_close(h, expected, rtol=0, atol=0)
    torch.testing.assert_close(outputs[1].cpu(), initial, rtol=0, atol=0)

"""NPU regressions for ChunkFwdH's chunk-first output and KDA V2 recompute.

Run with the wheel built from this checkout, on Ascend950 for the KDA cases.
The FwdH oracle is the existing independent ATK CPU implementation.
"""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

pytest.importorskip("torch_npu")
from fla_npu.ops.ascendc import chunk_fwd_h, chunk_kda_bwd, chunk_kda_fwd


@pytest.fixture(scope="module")
def fwd_h_reference():
    pytest.importorskip("atk")
    path = Path(__file__).resolve().parents[1] / "atk/chunk_fwd_h/executor_chunk_fwd_h.py"
    spec = importlib.util.spec_from_file_location("fwd_h_layout_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_outputs(actual, expected):
    assert len(actual) == len(expected)
    for got, ref in zip(actual, expected):
        if ref is None:
            assert got is None
            continue
        assert got.shape == ref.shape
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got.cpu().float(), ref.cpu().float(), rtol=0.02, atol=0.002)


@pytest.mark.parametrize("seed", [20260919, 20260920, 20260921])
@pytest.mark.parametrize("state_v_first", [False, True])
@pytest.mark.parametrize("mode,b,hk,hv,t,seqlens", [
    ("g", 2, 2, 6, 129, None),
    ("gk", 1, 3, 3, 129, None),  # H == number of chunks
    ("g", 1, 2, 6, 194, [1, 64, 129]),
    ("gk", 1, 3, 3, 194, [1, 64, 129]),
])
def test_fwd_h_chunk_first(fwd_h_reference, seed, state_v_first, mode, b, hk, hv, t, seqlens):
    spec = dict(B=b, HK=hk, HV=hv, T=t, seed=seed, mode=mode,
                gate_dtype="fp32", state_dtype="fp32", output_final_state=True,
                state_v_first=state_v_first, use_exp2=True, seqlens=seqlens,
                explicit_chunk_indices=seqlens is not None)
    cpu = fwd_h_reference.build_inputs(spec, torch.device("cpu"))
    expected = fwd_h_reference.run_cpu(spec, cpu)
    move = lambda value: value.npu() if value is not None else None
    kwargs = dict(g=move(cpu.g), gk=move(cpu.gk), initial_state=move(cpu.initial_state),
                  output_final_state=True, cu_seqlens=cpu.cu_seqlens,
                  chunk_indices=cpu.chunk_indices, use_exp2=True, state_v_first=state_v_first)
    inputs = tuple(move(value) for value in (cpu.k, cpu.w, cpu.u))
    actual = chunk_fwd_h(*inputs, **kwargs)
    torch.npu.synchronize()
    chunks = sum((n + 63) // 64 for n in seqlens) if seqlens else (t + 63) // 64
    expected_shape = (chunks, hv, 128, 128) if seqlens else (b, chunks, hv, 128, 128)
    assert actual[0].shape == expected_shape
    assert actual[0].is_contiguous()
    assert_outputs(actual, expected)
    again = chunk_fwd_h(*inputs, **kwargs)
    torch.npu.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(actual, again))


@pytest.mark.parametrize("b,h,t,seqlens", [
    (1, 8, 128, None),
    (1, 8, 512, None),
    (2, 8, 192, None),
    (1, 8, 384, [64, 128, 192]),
    (1, 8, 512, [64, 192, 256]),
    (1, 64, 4096, None),
])
def test_kda_v2_recompute_matches_saved(b, h, t, seqlens):
    if "950" not in torch.npu.get_device_name(torch.npu.current_device()):
        pytest.skip("KDA V2 requires Ascend950")
    generator = torch.Generator().manual_seed(20260920)
    shape = (h, t, 128) if seqlens else (b, h, t, 128)
    q, k = [(torch.randn(shape, generator=generator) * 0.05).bfloat16().npu() for _ in range(2)]
    v = (torch.randn(shape, generator=generator) * 0.1).bfloat16().npu()
    raw_g = (torch.randn(shape, generator=generator) * 0.2).npu()
    beta = torch.sigmoid(torch.randn(shape[:-1], generator=generator)).npu()
    a_log = (torch.randn(h, generator=generator) * 0.1).npu()
    bias = (torch.randn(h, 128, generator=generator) * 0.1).npu()
    d_o = (torch.randn(shape, generator=generator) * 0.1).bfloat16().npu()
    cu = None
    if seqlens:
        cu = [0]
        for length in seqlens:
            cu.append(cu[-1] + length)
    common = dict(scale=128 ** -0.5, chunk_size=64, cu_seqlens=cu,
                  safe_gate=True, lower_bound=-1.0, use_gate_in_kernel=True,
                  A_log=a_log, dt_bias=bias, use_exp2=True, state_v_first=False)
    forward = chunk_kda_fwd(q, k, v, raw_g, beta, layout="NTD" if cu else "BNSD",
                            disable_recompute=True, **dict(common, dt_bias=bias.flatten()))
    torch.npu.synchronize()
    _, _, gk, aqk, akk, w, _, qg, kg, v_new, h_saved, _ = forward
    kwargs = dict(q=q, k=k, v=v, beta=beta, Aqk=aqk, Akk=akk, d_o=d_o,
                  raw_g=raw_g, implementation="optimized", **common)
    saved = chunk_kda_bwd(gk=gk, w=w, qg=qg, kg=kg, v_new=v_new, h=h_saved,
                          disable_recompute=True, **kwargs)
    torch.npu.synchronize()
    recomputed = chunk_kda_bwd(gk=None, w=None, qg=None, kg=None, v_new=None, h=None,
                               disable_recompute=False, **kwargs)
    torch.npu.synchronize()
    assert_outputs(recomputed, saved)
    again = chunk_kda_bwd(gk=None, w=None, qg=None, kg=None, v_new=None, h=None,
                          disable_recompute=False, **kwargs)
    torch.npu.synchronize()
    for first, second in zip(recomputed, again):
        assert (first is None and second is None) or torch.equal(first, second)

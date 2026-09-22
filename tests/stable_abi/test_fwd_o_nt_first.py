"""NT-first FwdO regression using the existing FP64/low-precision oracle."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

pytest.importorskip("torch_npu")
from fla_npu.ops.ascendc import chunk_fwd_o


@pytest.fixture(scope="module")
def reference():
    path = Path(__file__).resolve().parents[2] / "torch_custom/fla_npu/test/test_npu_fwd_o_generalization.py"
    spec = importlib.util.spec_from_file_location("fwd_o_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shape", [
    (2, 2, 6, 129, 128, 64, None),
    (2, 3, 3, 129, 128, 64, None),  # HV == NT: shape cannot detect a wrong offset.
    (1, 2, 4, 194, 128, 64, (0, 1, 65, 194)),
    (1, 2, 4, 257, 256, 128, None),
])
@pytest.mark.parametrize("layout,state_v_first", [
    ("BNSD", False), ("NTD", False), ("BSND", False), ("TND", True),
])
def test_fwd_o_nt_first(reference, shape, layout, state_v_first):
    b, hk, hv, t, vdim, chunk, cu = shape
    optimized = layout in ("BSND", "TND")
    if optimized and ("950" not in torch.npu.get_device_name(torch.npu.current_device()) or vdim != 128 or chunk != 64):
        pytest.skip("BSND/TND and state_v_first require the A5 optimized domain")
    if layout in ("NTD", "TND") and b != 1:
        pytest.skip("rank-3 output requires B=1")
    case = reference.FwdOCase("nt_first", b, hk, hv, t, vdim, chunk, cu)
    nt = sum((y - x + chunk - 1) // chunk for x, y in zip(cu, cu[1:])) if cu else (t + chunk - 1) // chunk
    torch.manual_seed(20260921)
    q = torch.randn(b, hk, t, 128).bfloat16() * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn(b, hv, t, vdim).bfloat16() * 0.1
    h_old = torch.randn(b, hv, nt, 128, vdim).bfloat16() * 0.1
    g = torch.zeros(b, hv, t)
    scale = 128 ** -0.5
    golden = reference._reference(case, q, k, v, h_old, g, scale, False)
    benchmark = reference._reference(case, q, k, v, h_old, g, scale, True)
    h = h_old.transpose(1, 2)
    if cu is not None:
        h = h.squeeze(0)
    if state_v_first:
        h = h.transpose(-1, -2)
    args = [x.npu() for x in (q, k, v, h.contiguous())]
    kwargs = dict(g=g.npu(), chunk_size=chunk, output_layout=layout,
                  transpose_state_layout=state_v_first,
                  cu_seqlens=list(cu) if cu else None,
                  chunk_indices=reference._chunk_indices(cu, chunk) if cu else None)
    actual = chunk_fwd_o(*args, scale, **kwargs)
    torch.npu.synchronize()
    repeated = chunk_fwd_o(*args, scale, **kwargs)
    torch.npu.synchronize()
    assert torch.equal(actual, repeated)
    if layout == "BSND":
        actual = actual.transpose(1, 2)
    elif layout == "TND":
        actual = actual.transpose(0, 1).unsqueeze(0)
    elif layout == "NTD":
        actual = actual.unsqueeze(0)
    assert actual.shape == golden.shape
    result = reference.ct.dual(actual.cpu().float(), golden.float(), benchmark.float(), level="L1")
    assert result["success"], result


def test_fwd_o_rejects_head_first():
    q = torch.zeros(1, 2, 129, 128, dtype=torch.bfloat16, device="npu")
    h_old = torch.zeros(1, 2, 3, 128, 128, dtype=q.dtype, device="npu")
    with pytest.raises((RuntimeError, ValueError)):
        chunk_fwd_o(q, q, q, h_old, 128 ** -0.5,
                    g=torch.zeros(1, 2, 129, device="npu"), chunk_size=64)

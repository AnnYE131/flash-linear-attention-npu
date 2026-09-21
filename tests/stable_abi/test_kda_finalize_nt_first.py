"""NT-first Finalize against an independent per-head FP64 formula.

Run against the installed wheel, once per Stable ABI/ctypes backend.
"""

import pytest
import torch

pytest.importorskip("torch_npu")
from fla_npu.ops.ascendc import chunk_kda_fwd_finalize


def reference(q, a, v, h, layout, state_v_first, lengths):
    packed = q.ndim == 3
    if packed:
        q, a = q.unsqueeze(0), a.unsqueeze(0)
    if v.ndim == 3:
        v = v.unsqueeze(0)
    b, heads, tokens, _ = q.shape
    out = torch.empty((b, heads, tokens, 128), dtype=torch.float64)
    for batch in range(b):
        offset, chunk_id = 0, 0
        for length in lengths:
            for local in range(0, length, 64):
                begin, end = offset + local, offset + min(local + 64, length)
                for head in range(heads):
                    state = h[batch, chunk_id, head].double()
                    if state_v_first:
                        state = state.T
                    out[batch, head, begin:end] = (
                        q[batch, head, begin:end].double() @ state
                        + a[batch, head, begin:end, :end-begin].double()
                        @ v[batch, head, begin:end].double()
                    )
                chunk_id += 1
            offset += length
    if layout in ("BSND", "TND"):
        out = out.transpose(1, 2)
    return (out[0] if packed else out).contiguous()


@pytest.mark.parametrize("seed", [20260921, 20260922, 20260923])
@pytest.mark.parametrize("layout", ["BNSD", "BSND", "NTD", "TND"])
@pytest.mark.parametrize("svf", [False, True])
@pytest.mark.parametrize("lengths,heads", [([129], 3), ([129], 5), ([1, 64, 65], 3)])
def test_finalize_nt_first(seed, layout, svf, lengths, heads):
    packed = layout in ("NTD", "TND")
    variable = len(lengths) > 1
    batch = 1 if packed or variable else 2
    tokens = sum(lengths)
    chunks = sum((length + 63) // 64 for length in lengths)
    gen = torch.Generator().manual_seed(seed)

    def rand(shape):
        return (torch.rand(shape, generator=gen) * 2 - 1).bfloat16()

    shape = (heads, tokens) if packed else (batch, heads, tokens)
    q, a = rand((*shape, 128)), rand((*shape, 64))
    v = rand((batch, heads, tokens, 128))
    h = rand((batch, chunks, heads, 128, 128))
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    expected = reference(q, a, v, h, layout, svf, lengths)
    inputs = [x.npu() for x in (q, a, v, h)]
    kwargs = dict(output_layout=layout, state_v_first=svf,
                  cu_seqlens=cu if variable else None)
    result = chunk_kda_fwd_finalize(*inputs, **kwargs)
    torch.npu.synchronize()
    assert result.shape == expected.shape
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.cpu().float(), expected.bfloat16().float(),
                               rtol=0.02, atol=0.002)
    repeat = chunk_kda_fwd_finalize(*inputs, **kwargs)
    torch.npu.synchronize()
    assert torch.equal(result, repeat)
    if seed == 20260921 and heads != chunks:
        strided_h = inputs[3].transpose(1, 2).contiguous().transpose(1, 2)
        strided_result = chunk_kda_fwd_finalize(*inputs[:3], strided_h, **kwargs)
        torch.npu.synchronize()
        assert torch.equal(result, strided_result)
    if heads == chunks:
        wrong = reference(q, a, v, h.transpose(1, 2).contiguous(), layout, svf, lengths)
        assert not torch.allclose(wrong, expected, rtol=0.02, atol=0.002)


@pytest.mark.parametrize("bad", ["head_first", "rank4", "empty_sequence", "noncanonical", "k64", "fp32"])
def test_finalize_rejects_invalid_contract(bad):
    q = torch.zeros((1, 5, 129, 128), dtype=torch.bfloat16, device="npu")
    a = torch.zeros((1, 5, 129, 64), dtype=torch.bfloat16, device="npu")
    h = torch.zeros((1, 3, 5, 128, 128), dtype=torch.bfloat16, device="npu")
    v = q.clone()
    kwargs = dict(output_layout="BNSD")
    if bad == "head_first":
        h = h.transpose(1, 2).contiguous()
    elif bad == "rank4":
        h = h.squeeze(0)
    elif bad == "empty_sequence":
        kwargs["cu_seqlens"] = [0, 0, 129]
    elif bad == "noncanonical":
        kwargs.update(cu_seqlens=[0, 129], chunk_indices=[0, 1, 0, 0, 0, 2])
    elif bad == "k64":
        q = q[..., :64].contiguous()
    elif bad == "fp32":
        h = h.float()
    with pytest.raises((RuntimeError, ValueError)):
        chunk_kda_fwd_finalize(q, a, v, h, **kwargs)
        torch.npu.synchronize()

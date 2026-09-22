"""Direct ctypes -> ACLNN state descriptor rejection checks (installed wheel).

The ctypes FwdO adapter allocates outputs but does not validate h;
these errors must come from the compiled host before launching the kernel.
"""
import pytest
import torch

pytest.importorskip("torch_npu")
from fla_npu.ops.ascendc import _aclnn_ctypes


@pytest.mark.parametrize("case", ["valid", "rank", "short", "extra", "heads"])
def test_fwd_o_rejects_invalid_state_schedule(case):
    q = torch.zeros(1, 2, 130, 128, dtype=torch.bfloat16, device="npu")
    g = torch.zeros(1, 2, 130, device="npu")
    shape = (4, 2, 128, 128)
    cu, ci = [0, 1, 65, 130], [0, 0, 1, 0, 2, 0, 2, 1]
    if case == "rank": shape = (1, *shape)
    if case == "short": shape = (3, *shape[1:])
    if case == "extra": shape = (5, *shape[1:])
    if case == "heads": shape = (4, 3, 128, 128)
    h = torch.zeros(shape, dtype=q.dtype, device="npu")
    if case == "valid":
        output = _aclnn_ctypes.npu_chunk_fwd_o(q, q, q, h, 1., g=g,
            cu_seqlens=cu, chunk_indices=ci, chunk_size=64, output_layout="BNSD")
        torch.npu.synchronize()
        assert torch.equal(output.cpu(), torch.zeros_like(output.cpu()))
        return
    with pytest.raises(RuntimeError):
        _aclnn_ctypes.npu_chunk_fwd_o(q, q, q, h, 1., g=g,
            cu_seqlens=cu, chunk_indices=ci, chunk_size=64, output_layout="BNSD")

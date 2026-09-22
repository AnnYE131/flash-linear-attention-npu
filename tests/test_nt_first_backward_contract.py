"""P3 CPU allocation and independent autograd checks; no device acceptance."""
import ctypes
import itertools
import unittest

import torch

from test_nt_first_cpu_references import load_reference
from test_nt_first_forward_contract import OPS, load_functions


class BackwardContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        runtime = {}
        names = {"shape", "empty", "empty_like", "optional_bool", "chunk_num"}
        load_functions(OPS / "_runtime.py", names, runtime)
        ns = {"ctypes": ctypes, **{"_" + n: runtime[n] for n in names}}
        ns["_call_aclnn"] = lambda name, args, outputs: outputs
        load_functions(OPS / "_aclnn_ctypes.py", {"npu_chunk_gated_delta_rule_bwd_dhu"}, ns)
        cls.ops = ns
        cls.ref = load_reference("torch_custom/fla_npu/test/test_bwd_dhu.py")

    def test_public_allocations(self):
        for packed, svf, heads, vdim in itertools.product((False, True), (False, True), (2, 3), (128, 256)):
            with self.subTest(packed=packed, svf=svf, heads=heads, vdim=vdim):
                b, t = (1, 130) if packed else (2, 129)
                q = torch.empty(b, 1, t, 128, dtype=torch.bfloat16)
                w = torch.empty(b, heads, t, 128, dtype=q.dtype)
                dv = torch.empty(b, heads, t, vdim, dtype=q.dtype)
                tail = (vdim, 128) if svf else (128, vdim)
                initial = torch.empty(3 if packed else b, heads, *tail, dtype=q.dtype)
                result = self.ops["npu_chunk_gated_delta_rule_bwd_dhu"](
                    q, q, w, dv, dv, 1., 64, g=torch.zeros(b, heads, t), h0=initial,
                    cu_seqlens=[0, 1, 65, 130] if packed else None,
                    chunk_indices=[0, 0, 1, 0, 2, 0, 2, 1] if packed else None,
                    transpose_state_layout=svf)
                self.assertEqual(result[0].shape, (4, heads, *tail) if packed else (b, 3, heads, *tail))
                self.assertEqual(result[1].shape, initial.shape)
                self.assertEqual(result[2].shape, dv.shape)
                self.assertTrue(all(x.dtype == q.dtype for x in result))

    def test_state_gradient_against_autograd(self):
        # Independent forward recurrence, differentiated by PyTorch. Small K/V
        # verify reference mathematics only, not device support for these sizes.
        for packed, svf in itertools.product((False, True), repeat=2):
            with self.subTest(packed=packed, svf=svf):
                torch.manual_seed(701)
                b, t, heads, kd, vd, bt = (1 if packed else 2), 5, 2, 3, 4, 2
                cu = [0, 1, 4, 5] if packed else None
                ci = [0, 0, 1, 0, 1, 1, 2, 0] if packed else None
                def random(*shape):
                    return torch.randn(*shape, dtype=torch.float64) * .1
                q, k = random(b, 1, t, kd), random(b, 1, t, kd)
                w, do, dv = random(b, heads, t, kd), random(b, heads, t, vd), random(b, heads, t, vd)
                initial = random(3 if packed else b, heads, kd, vd).requires_grad_()
                u = random(b, heads, t, vd).requires_grad_()
                saved = []
                loss = torch.zeros((), dtype=torch.float64)
                for batch in range(b):
                    intervals = list(zip(cu, cu[1:])) if packed else [(0, t)]
                    for seq, (left, right) in enumerate(intervals):
                        state = initial[seq if packed else batch]
                        for start in range(left, right, bt):
                            end = min(start + bt, right)
                            values = u[batch, :, start:end] - w[batch, :, start:end] @ state
                            output = q[batch, :, start:end] @ state * .7
                            loss = loss + (output * do[batch, :, start:end]).sum()
                            loss = loss + (values * dv[batch, :, start:end]).sum()
                            state = state + k[batch, :, start:end].transpose(-1, -2) @ values
                            state.retain_grad()
                            saved.append(state)
                        loss = loss + state.sum() * 0  # Zero terminal-state gradient.
                loss.backward()
                physical = lambda x: x.transpose(-1, -2).contiguous() if svf else x
                actual = self.ref.chunk_gated_delta_rule_bwd_dhu_cpu(
                    q, k, w, do, dv, cu_seqlens=cu, chunk_indices=ci,
                    h0=physical(initial.detach()), dht=None,
                    scale=.7, chunk_size=bt, golden_mode="fp64", nt_first=True, state_v_first=svf)
                expected_dh = torch.stack([state.grad for state in saved])
                if not packed:
                    expected_dh = expected_dh.reshape(b, 3, heads, kd, vd)
                for result, expected in zip(actual, (physical(expected_dh), physical(initial.grad), u.grad)):
                    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

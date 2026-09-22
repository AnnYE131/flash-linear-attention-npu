"""CPU-only P2 regressions. Actual Python allocation/reference code, no ACLNN launch.

This does not validate C++ descriptors, tiling, or device results.
Run: python tests/test_nt_first_forward_contract.py
"""
from __future__ import annotations

import ast
import ctypes
import itertools
from pathlib import Path
import unittest

import torch

from test_nt_first_cpu_references import load_reference

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "torch_custom/fla_npu/fla_npu/ops/ascendc"


def load_functions(path, names, namespace):
    nodes = ast.parse("from __future__ import annotations").body
    nodes += [node for node in ast.parse(path.read_text(encoding="utf-8")).body
              if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


class ForwardContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        runtime = {}
        names = {"shape", "empty", "empty_like", "optional_bool", "optional_int", "optional_float"}
        load_functions(OPS / "_runtime.py", names, runtime)
        ns = {"ctypes": ctypes, **{"_" + name: runtime[name] for name in names}}
        # Stop at the launch boundary: allocation and validation execute unchanged.
        ns["_call_aclnn"] = lambda name, args, outputs: outputs
        load_functions(OPS / "_aclnn_ctypes.py", {
            "npu_chunk_fwd_h", "npu_chunk_gated_delta_rule_fwd", "npu_chunk_gated_delta_rule_fwd_h",
            "_kda_total_chunks", "_kda_build_chunk_indices", "_kda_ceil_div",
            "_chunk_fwd_h_ceil_div", "_chunk_fwd_h_build_chunk_indices", "_chunk_fwd_h_total_chunks",
        }, ns)
        cls.ops = ns
        cls.finalize = load_reference(
            "tests/atk/chunk_kda_fwd_finalize/executor_chunk_kda_fwd_finalize.py")
        cls.shared = load_reference("tests/atk/chunk_fwd_h/executor_chunk_fwd_h.py")

    def test_gdn_export_allocations(self):
        for packed, svf, heads, vdim in itertools.product((False, True), (False, True), (2, 3), (128, 256)):
            with self.subTest(packed=packed, svf=svf, heads=heads, vdim=vdim):
                b, t = (1, 130) if packed else (2, 129)
                q = torch.empty(b, 1, t, 128, dtype=torch.bfloat16)
                v = torch.empty(b, heads, t, vdim, dtype=q.dtype)
                g = torch.empty(b, t, heads)
                cu, ci = ([0, 1, 65, 130], [0, 0, 1, 0, 2, 0, 2, 1]) if packed else (None, None)
                result = self.ops["npu_chunk_gated_delta_rule_fwd"](
                    q, q, v, g, g, cu_seqlens=cu, chunk_indices=ci,
                    state_v_first=svf, return_intermediate_states=True, output_final_state=True)
                tail = (vdim, 128) if svf else (128, vdim)
                self.assertEqual(result[5].shape, (4, heads, *tail) if packed else (b, 3, heads, *tail))
                self.assertEqual(result[1].shape, (3 if packed else b, heads, *tail))
                self.assertEqual(result[5].dtype, q.dtype)

    def test_shared_fwd_h_allocations(self):
        for packed, svf, name in itertools.product((False, True), (False, True),
                ("npu_chunk_fwd_h", "npu_chunk_gated_delta_rule_fwd_h")):
            with self.subTest(packed=packed, svf=svf, name=name):
                b, t = (1, 130) if packed else (2, 129)
                k = torch.empty(b, 2, t, 128, dtype=torch.bfloat16)
                g = torch.empty(b, 2, t)
                result = self.ops[name](
                    k, k, k, g=g, cu_seqlens=[0, 1, 65, 130] if packed else None,
                    state_v_first=svf, output_final_state=True)
                self.assertEqual(result[0].shape, (4, 2, 128, 128) if packed else (b, 3, 2, 128, 128))
                self.assertEqual(result[1].shape, k.shape)
                self.assertEqual(result[2].shape, (3 if packed else b, 2, 128, 128))

    def test_finalize_rank_and_capacity(self):
        for packed, rank3, svf in itertools.product((False, True), repeat=3):
            with self.subTest(packed=packed, rank3=rank3, svf=svf):
                t, chunks = (130, 4) if packed else (129, 3)
                shape = (2, t) if rank3 else (1, 2, t)
                q = torch.empty(*shape, 128, dtype=torch.bfloat16)
                a = torch.empty(*shape, 64, dtype=q.dtype)
                h = torch.empty((chunks, 2, 128, 128) if packed else (1, chunks, 2, 128, 128), dtype=q.dtype)
                kwargs = dict(cu_seqlens=[0, 1, 65, 130] if packed else None,
                              output_layout="NTD" if rank3 else "BNSD", state_v_first=svf)
                def prepare(state):
                    return self.finalize._prepare(dict(qg_scaled=q, aqk=a, v_new=q,
                                                       h=state, **kwargs))
                prepared = prepare(h)
                self.assertEqual(prepared.h.shape, h.shape)
                bad_rank = h.unsqueeze(0) if packed else h.squeeze(0)
                bad_count = h[:-1] if packed else h[:, :-1]
                for bad in (bad_rank, bad_count):
                    with self.assertRaises(ValueError):
                        prepare(bad)

    def test_shared_h_to_finalize_reference(self):
        # Three unequal sequences need four states, not ceil(130/64)=3.
        for svf in (False, True):
            with self.subTest(svf=svf):
                spec = dict(B=1, HK=2, HV=2, T=130, seed=20260922, mode="gk",
                            gate_dtype="fp32", state_dtype="fp32", output_final_state=True,
                            state_v_first=svf, use_exp2=True, seqlens=[1, 64, 65])
                inputs = self.shared.build_inputs(spec, torch.device("cpu"))
                h, v, _ = self.shared.run_cpu(spec, inputs)
                self.assertEqual(h.shape, (4, 2, 128, 128))
                q = inputs.k
                a = torch.zeros(1, 2, 130, 64, dtype=torch.bfloat16)
                prepared = self.finalize._prepare(dict(qg_scaled=q, aqk=a, v_new=v,
                    h=h, cu_seqlens=[0, 1, 65, 130], chunk_indices=None,
                    output_layout="BNSD", state_v_first=svf))
                result = self.finalize.run_cpu(prepared)
                expected = torch.empty_like(result)
                for c, (left, right) in enumerate(((0, 1), (1, 65), (65, 129), (129, 130))):
                    for head in range(2):
                        state = h[c, head].double()
                        if svf:
                            state = state.T
                        expected[0, head, left:right] = q[0, head, left:right].double() @ state
                torch.testing.assert_close(result, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

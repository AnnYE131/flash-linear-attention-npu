"""CPU-only layout equivalence checks for the existing mathematical references.

Executors import ATK and legacy PTA imports torch_npu at module import time.
Load only their Python definitions/constants here: no mocked tensor operations,
no NPU calls, and no rewritten reference math. This is not ATK acceptance.
Run: python tests/test_nt_first_cpu_references.py
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import types
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_reference(relative, extra=None):
    path = ROOT / relative
    name = "cpu_layout_" + path.stem
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    if extra:
        module.__dict__.update(extra)
    nodes = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ImportFrom) and node.module in {
            "__future__", "typing", "dataclasses"
        }:
            nodes.append(node)
        elif isinstance(node, ast.Import) and all(
            item.name in {"math", "torch"} for item in node.names
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef):
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in {"PreparedInputs", "Inputs"}:
            nodes.append(node)
        elif (isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets)
              and isinstance(node.value, (ast.Constant, ast.Dict))):
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def nt_state(old, *, packed=False, v_first=False):
    result = old.transpose(1, 2)
    if packed:
        result = result.squeeze(0)
    if v_first:
        result = result.transpose(-1, -2)
    return result.contiguous()


class LayoutReferences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        common = load_reference("tests/atk/common/_ascendc_common_executor.py")
        cls.fwd = load_reference(
            "tests/atk/chunk_gated_delta_rule_fwd_h/executor_chunk_gated_delta_rule_fwd_h.py",
            {"_num_chunks": common._num_chunks, "_chunks": common._chunks},
        )
        cls.shared = load_reference("tests/atk/chunk_fwd_h/executor_chunk_fwd_h.py")
        cls.dhu = load_reference("torch_custom/fla_npu/test/test_bwd_dhu.py")
        cls.finalize = load_reference(
            "tests/atk/chunk_gated_delta_rule_bwd_finalize/scripts/chunk_gated_delta_rule_bwd_finalize_cpu.py"
        )
        inner = load_reference("tests/atk/chunk_bwd_dqkwg/scripts/chunk_bwd_dqkwg_cpu.py")
        cls.dqkwg = load_reference(
            "tests/atk/chunk_bwd_dqkwg/executor_chunk_bwd_dqkwg.py",
            {"chunk_bwd_dqkwg_cpu": inner.chunk_bwd_dqkwg_cpu},
        )

    def equal(self, left, right):
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            if a is None or b is None:
                self.assertIs(a, b)
            else:
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(a.dtype, b.dtype)
                self.assertTrue(torch.isfinite(a).all())
                self.assertTrue(torch.equal(a, b))

    def test_legacy_fwd_h_dense(self):
        for seed in (20260919, 20260920, 20260921):
            for heads in (2, 3):  # NT=3: unequal and equal head/chunk axes.
                with self.subTest(seed=seed, heads=heads):
                    torch.manual_seed(seed)
                    inputs = dict(k=torch.randn(2, 1, 5, 3) * .1,
                                  w=torch.randn(2, heads, 5, 3) * .1,
                                  u=torch.randn(2, heads, 5, 2) * .1,
                                  g=torch.randn(2, heads, 5) * .01, chunk_size=2)
                    old = self.fwd._forward_h_ref(inputs)
                    new = self.fwd._forward_h_ref(inputs, nt_first=True)
                    self.equal(new, (nt_state(old[0]), old[1]))

    def test_dhu_dense_packed_and_state_order(self):
        for seed in (20260919, 20260920, 20260921):
            for packed in (False, True):
                for v_first in (False, True):
                    with self.subTest(seed=seed, packed=packed, v_first=v_first):
                        torch.manual_seed(seed)
                        b, hv, t, k, v = (1 if packed else 2), 3, 5, 3, 2
                        q, key = [torch.randn(b, 1, t, k).double() * .1 for _ in range(2)]
                        w = torch.randn(b, hv, t, k).double() * .1
                        do, dv = [torch.randn(b, hv, t, v).double() * .1 for _ in range(2)]
                        kw = dict(chunk_size=2, golden_mode="fp64", state_v_first=v_first,
                                  h0=torch.ones((3 if packed else b), hv, *((v, k) if v_first else (k, v))))
                        if packed:
                            # sum(ceil(length/2))=4, ceil(total_tokens/2)=3.
                            kw.update(cu_seqlens=[0, 1, 4, 5], chunk_indices=[0, 0, 1, 0, 1, 1, 2, 0])
                        old = self.dhu.chunk_gated_delta_rule_bwd_dhu_cpu(q, key, w, do, dv, **kw)
                        new = self.dhu.chunk_gated_delta_rule_bwd_dhu_cpu(q, key, w, do, dv, nt_first=True, **kw)
                        self.equal(new, (nt_state(old[0], packed=packed, v_first=v_first), *old[1:]))

    def test_shared_fwd_h_packed_rank(self):
        for v_first in (False, True):
            with self.subTest(v_first=v_first):
                spec = dict(B=1, HK=1, HV=2, T=130, seed=20260919, mode="g",
                            gate_dtype="fp32", state_dtype="fp32", output_final_state=True,
                            state_v_first=v_first, use_exp2=True, seqlens=[1, 64, 65],
                            explicit_chunk_indices=True)
                inputs = self.shared.build_inputs(spec, torch.device("cpu"))
                kw = dict(output_final_state=True, use_exp2=True, state_v_first=v_first)
                old = self.shared._reference(inputs, **kw)
                new = self.shared._reference(inputs, packed_output=True, **kw)
                self.assertEqual(new[0].shape, (4, 2, 128, 128))
                self.equal(new, (old[0].squeeze(0), *old[1:]))

    def test_finalize_consumes_same_states(self):
        for packed in (False, True):
            for v_first in (False, True):
                with self.subTest(packed=packed, v_first=v_first):
                    torch.manual_seed(20260919)
                    b, hv, t, nt = (1 if packed else 2), 2, (66 if packed else 65), (3 if packed else 2)
                    q, k = [torch.randn(b, 1, t, 128) * .01 for _ in range(2)]
                    v, vn, do, du = [torch.randn(b, hv, t, 128) * .01 for _ in range(4)]
                    g = torch.randn(b, hv, t) * .01
                    beta = torch.rand(b, hv, t)
                    h, dh = [torch.randn(b, hv, nt, 128, 128) * .01 for _ in range(2)]
                    a = torch.randn(b, hv, t, 64) * .01
                    kw = dict(state_v_first=v_first)
                    if packed:
                        # lengths 1,65 => 3 chunks, ceil(66/64)=2.
                        kw.update(cu_seqlens=[0, 1, 66], chunk_indices=[0, 0, 1, 0, 1, 1])
                    fn = self.finalize.chunk_gated_delta_rule_bwd_finalize_golden
                    old = fn(q, k, v, vn, do, du, g, beta, h, dh, a, **kw)
                    new = fn(q, k, v, vn, do, du, g, beta,
                             nt_state(h, packed=packed), nt_state(dh, packed=packed), a,
                             nt_first=True, **kw)
                    self.equal(new, old)
                    if not packed:
                        wrong = fn(q, k, v, vn, do, du, g, beta, h, dh, a,
                                   nt_first=True, **kw)
                        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(wrong, new)))

    def test_dqkwg_consumes_same_states(self):
        for packed in (False, True):
            for v_first in (False, True):
                with self.subTest(packed=packed, v_first=v_first):
                    torch.manual_seed(20260920)
                    b, hv, t, nt = (1 if packed else 2), 3, 5, 3
                    q, k = [torch.randn(b, 1, t, 3).double() * .1 for _ in range(2)]
                    v, do, dv = [torch.randn(b, hv, t, 2).double() * .1 for _ in range(3)]
                    g = torch.randn(b, hv, t).double() * .01
                    h, dh = [torch.randn(b, hv, nt, 3, 2).double() * .1 for _ in range(2)]
                    cu = [0, 1, 5] if packed else None
                    fn = self.dqkwg.chunk_bwd_dqkwg_torch
                    old = fn(q, k, v, do, h, dh, None, g, dv, 1., cu, 2)
                    new = fn(q, k, v, do, nt_state(h, packed=packed, v_first=v_first),
                             nt_state(dh, packed=packed, v_first=v_first), None, g, dv, 1., cu, 2,
                             nt_first=True, state_v_first=v_first)
                    self.equal(new, old)

    def test_equal_axes_still_detect_wrong_content(self):
        # Shape equality is insufficient; head and chunk carry different values.
        h = torch.arange(2 * 2 * 3 * 2).reshape(1, 2, 2, 3, 2)
        self.assertEqual(h.shape, nt_state(h).shape)
        self.assertFalse(torch.equal(h, nt_state(h)))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""不依赖 NPU 的模式、冻结矩阵与 executor 调用合同验证。"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from prepare_output_mode_cases import derive_cases

OP_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("output_contract", OP_DIR / "output_contract.py")
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)


class OutputModesTest(unittest.TestCase):
    def test_each_role_mode_state_combination(self):
        o, state, g, a = object(), object(), object(), object()
        for final in (False, True):
            case = SimpleNamespace(output_final_state=final)
            for disabled in (False, True):
                names = ("o", "final_state") if final else ("o",)
                expected = (o, state) if final else (o,)
                if not disabled:
                    names += ("g_cumsum", "A")
                    expected += (g, a)
                self.assertEqual(CONTRACT.comparison_output_names(case, disabled), names)
                for role in ("dut", "benchmark", "golden"):
                    with self.subTest(final=final, disabled=disabled, role=role):
                        aux = (None, None) if role == "dut" and disabled else (g, a)
                        raw = (o, state if final else None, *aux)
                        if role == "golden" and not final:
                            raw = (o, *aux)
                        result = CONTRACT.select_public_outputs(raw, case, role, disabled)
                        self.assertEqual(result, expected)

    def test_rejects_wrong_optional_outputs(self):
        case = SimpleNamespace(output_final_state=False)
        for raw, disabled in (((1, None, 2, 3), True), ((1, None, None, None), False),
                              ((1, 2, None, None), True), ((None, None, None, None), True)):
            with self.subTest(raw=raw, disabled=disabled), self.assertRaises(RuntimeError):
                CONTRACT.select_public_outputs(raw, case, "dut", disabled)
        with self.assertRaises(RuntimeError):
            CONTRACT.select_public_outputs((1, None, 2, 3),
                                           SimpleNamespace(output_final_state=True), "dut")
        for role in ("dut", "benchmark", "golden"):
            with self.assertRaises(RuntimeError):
                CONTRACT.select_public_outputs((1,), case, role)

    def test_derived_500_only_change_mode_and_seed(self):
        original = json.loads((OP_DIR / "atk_chunk_gated_delta_rule_fwd.json").read_bytes())
        before = json.dumps(original, sort_keys=True)
        self.assertEqual(len(original), 500)
        for mode in ("training", "inference"):
            for seed in (20260909, 20260910, 20260911):
                derived = derive_cases(original, mode, seed)
                self.assertEqual(len(derived), 500)
                for source, candidate in zip(original, derived):
                    self.assertEqual(candidate["default_seed"], seed)
                    attribute = candidate["inputs"].pop()
                    self.assertEqual(attribute["name"], "disable_recompute")
                    self.assertIs(attribute["range_values"], mode == "inference")
                    candidate["default_seed"] = source["default_seed"]
                    self.assertEqual(candidate, source)
        self.assertEqual(json.dumps(original, sort_keys=True), before)

    def test_rejects_already_derived_or_invalid_matrix(self):
        source = [{"id": 0, "inputs": [], "default_seed": None}]
        with self.assertRaises(ValueError):
            derive_cases(derive_cases(source, "training", 0), "inference", 1)
        for bad in ([], [{"id": 1, "inputs": []}]):
            with self.assertRaises(ValueError):
                derive_cases(bad, "training", 0)

    def test_real_run_npu_forwards_mode_only_to_dut(self):
        # 执行真实 run_npu 函数体，仅替换硬件依赖；不是 NPU 精度测试。
        tree = ast.parse((OP_DIR / "executor_chunk_gated_delta_rule_fwd.py").read_text(encoding="utf-8"))
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_npu")
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
        dut = Mock()
        six = Mock(return_value=(1, None, 2, 3))
        ascendc = SimpleNamespace(chunk_gated_delta_rule_fwd=dut)
        env = {"deterministic_initial_state": lambda case: None,
               "canonical_chunk_indices": lambda cu, size: None,
               "SIX_ACLNN_OPS": (), "run_six_aclnn_core": six,
               "select_public_outputs": CONTRACT.select_public_outputs}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "executor_run_npu", "exec"), env)
        case = SimpleNamespace(output_final_state=False, cu_seqlens=None, chunk_size=64, scale=0.125)
        with patch.dict(sys.modules, {"fla_npu.ops": SimpleNamespace(ascendc=ascendc)}):
            for disabled in (False, True):
                dut.return_value = (1, None, None, None) if disabled else (1, None, 2, 3)
                expected = (1,) if disabled else (1, 2, 3)
                self.assertEqual(env["run_npu"]("dut", range(5), case, disabled), expected)
                self.assertIs(dut.call_args.kwargs["disable_recompute"], disabled)
                self.assertEqual(env["run_npu"]("benchmark", range(5), case, disabled), expected)
                self.assertNotIn("disable_recompute", six.call_args.kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)

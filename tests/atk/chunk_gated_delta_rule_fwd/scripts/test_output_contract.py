#!/usr/bin/env python3
"""无需 NPU、ATK 或 torch 的输出布局合同检查。"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


OP_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gdn_output_contract", OP_DIR / "output_contract.py")
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)


class ArrayTensor:
    """用 NumPy 执行真实轴重排，模拟本合同需要的两个 Tensor 方法。"""

    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape

    def transpose(self, dim0, dim1):
        return ArrayTensor(self.values.swapaxes(dim0, dim1))

    def contiguous(self):
        return ArrayTensor(np.ascontiguousarray(self.values))


class OutputContractTest(unittest.TestCase):
    def setUp(self):
        self.case = SimpleNamespace(batch=2, tokens=5, v_heads=3, value_dim=7)
        self.head_first = np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)

    def test_reference_roles_reorder_values_to_public_layout(self):
        for role in ("golden", "benchmark"):
            with self.subTest(role=role):
                source = ArrayTensor(self.head_first.copy())
                result = CONTRACT.normalize_o_for_comparison(source, self.case, role)
                np.testing.assert_array_equal(result.values, self.head_first.swapaxes(1, 2))
                np.testing.assert_array_equal(source.values, self.head_first)
                self.assertTrue(result.values.flags.c_contiguous)

    def test_dut_is_not_transposed_again(self):
        source = ArrayTensor(self.head_first.swapaxes(1, 2).copy())
        self.assertIs(CONTRACT.normalize_o_for_comparison(source, self.case, "dut"), source)

    def test_equal_axes_do_not_hide_wrong_layout_inference(self):
        case = SimpleNamespace(batch=1, tokens=3, v_heads=3, value_dim=2)
        source = ArrayTensor(np.arange(18).reshape(1, 3, 3, 2))
        for role in ("benchmark", "golden"):
            with self.subTest(role=role):
                result = CONTRACT.normalize_o_for_comparison(source, case, role)
                np.testing.assert_array_equal(result.values, source.values.swapaxes(1, 2))
                self.assertFalse(np.array_equal(result.values, source.values))

    def test_rejects_wrong_source_shape_for_all_roles(self):
        for role in ("dut", "benchmark", "golden"):
            with self.subTest(role=role):
                with self.assertRaisesRegex(RuntimeError, "shape"):
                    CONTRACT.normalize_o_for_comparison(ArrayTensor(np.zeros((1,))), self.case, role)

    def test_rejects_unknown_role(self):
        with self.assertRaises(ValueError):
            CONTRACT.normalize_o_for_comparison(ArrayTensor(self.head_first), self.case, "unknown")

    def test_executor_adapts_only_after_cpu_return(self):
        tree = ast.parse((OP_DIR / "executor_chunk_gated_delta_rule_fwd.py").read_text(encoding="utf-8"))
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FunctionApi")
        method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "__call__")
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
        adapters = [node for node in calls if isinstance(node.func, ast.Name)
                    and node.func.id == "normalize_o_for_comparison"]
        self.assertEqual(len(adapters), 1)
        cpu_calls = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "cpu"]
        self.assertEqual(len(cpu_calls), 1)
        self.assertLess(cpu_calls[0].lineno, adapters[0].lineno)
        no_output = next(node for node in method.body if isinstance(node, ast.If)
                         and isinstance(node.test, ast.UnaryOp)
                         and isinstance(node.test.operand, ast.Name)
                         and node.test.operand.id == "with_output")
        self.assertLess(no_output.lineno, adapters[0].lineno)


if __name__ == "__main__":
    unittest.main(verbosity=2)

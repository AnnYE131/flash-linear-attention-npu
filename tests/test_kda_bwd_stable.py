"""Offline checks for KDA backward routing and the V2 host ABI."""
import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "torch_custom/fla_npu"
OPS = PACKAGE / "fla_npu/ops/ascendc"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KdaBackwardStableTest(unittest.TestCase):
    def test_routing_preserves_pr_contract(self):
        select = load(OPS / "_kda_policy.py")._select_kda_bwd_optimized
        self.assertFalse(select("auto", None, None, True))
        self.assertFalse(select("legacy", None, None, False))
        self.assertTrue(select("optimized", None, None, True))
        self.assertTrue(select("auto", None, None, False))
        self.assertTrue(select("auto", object(), object(), True))
        for args in (("bad", None, None, True),
                     ("auto", object(), None, True),
                     ("legacy", object(), object(), True)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                select(*args)

    def test_metadata_compacts_empty_sequences_and_checks_indices(self):
        metadata = load(OPS / "_kda_policy.py")._canonical_kda_bwd_metadata
        self.assertEqual(metadata([0, 1, 1, 65], [0, 0, 2, 0], 65),
                         ((0, 1, 65), (0, 0, 1, 0), 2))
        self.assertEqual(metadata(None, None, 65), (None, None, 2))
        with self.assertRaises(ValueError):
            metadata([0, 1, 65], [0, 0, 0, 1], 65)
        with self.assertRaises(ValueError):
            metadata(types.SimpleNamespace(device=types.SimpleNamespace(type="npu")),
                     None, 65)

    def test_v2_header_matches_both_launchers(self):
        tool = load(PACKAGE / "tools/op_abi_validate.py")
        symbol = "aclnnChunkKdaBwdV2"
        headers = tool.parse_headers(
            ROOT / "fla/ops/ascendc/kda/chunk_kda_bwd/op_host/op_api", {symbol})
        adapters = tool.adapter_calls(PACKAGE / "csrc/src")
        table = tool.parse_ctypes_table(OPS / "_aclnn_ctypes.py")
        self.assertEqual(tool.compare(headers[symbol], adapters[symbol][1]), [])
        self.assertEqual(tool.compare(headers[symbol], table[symbol]), [])

    def test_stable_dispatch_argument_order_and_stream(self):
        tree = ast.parse((OPS / "_stable.py").read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("_kda_bwd_single_launch", "_kda_bwd_optimized_launch")]
        namespace = {"__package__": "kda_test"}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "_stable.py", "exec"), namespace)
        names = ("q", "k", "v", "beta", "gk", "Aqk", "Akk", "w", "qg",
                 "kg", "v_new", "h", "d_o", "raw_g", "A_log", "q_rstd", "k_rstd")
        args = {name: object() for name in names}
        args.update(dt_bias=None, cu_seqlens=(0, 64), chunk_indices=(0, 0),
                    scale=0.125, chunk_size=64, safe_gate=True,
                    use_gate_in_kernel=True, lower_bound=-1.0,
                    disable_recompute=True, use_exp2=True, state_v_first=False)
        policy = types.ModuleType("kda_test._kda_policy")
        policy._prepare_kda_bwd_optimized = lambda value: value
        launch = Mock(return_value=(1, 2, 3, 4, 5, None, 7, None))
        namespace.update(_op=Mock(return_value=launch),
                         _host_ints=lambda x: ("host", x),
                         _current_stream_ptr=Mock(side_effect=[17, 23, 31]))
        with patch.dict(sys.modules, {"kda_test._kda_policy": policy}):
            namespace["_kda_bwd_optimized_launch"](args)
            namespace["_kda_bwd_optimized_launch"](args)
        call = launch.call_args_list[0].args
        expected = tuple(args[n] for n in names[:15]) + (
            None, ("host", (0, 64)), ("host", (0, 0)),
            0.125, 64, True, True, -1.0, True, True, False,
            True, args["q_rstd"], args["k_rstd"], 17)
        self.assertEqual(call, expected)
        self.assertEqual(launch.call_args_list[1].args[-1], 23)
        namespace["_kda_bwd_single_launch"](
            *(args[name] for name in names[:15]), None,
            0.125, 64, True, -1.0, True)
        self.assertEqual(launch.call_args.args[-4:], (False, None, None, 31))
        namespace["_op"].assert_called_with("npu_chunk_kda_bwd")


if __name__ == "__main__":
    unittest.main()

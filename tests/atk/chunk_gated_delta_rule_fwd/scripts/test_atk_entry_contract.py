#!/usr/bin/env python3
"""真实入口的参数传递与 DC 白名单单测；不替代 ATK/NPU 验收。"""

from __future__ import annotations

import ast
from enum import Enum
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

OP = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("atk_role_contract", OP / "atk_role_contract.py")
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)
DC_MODE = "ascend_use_deterministic_algorithms"


class AtkRoleTest(unittest.TestCase):
    def test_existing_roles_unchanged(self):
        for node in ("phase6", "npu_dut"):
            self.assertEqual(CONTRACT.role_for_atk_task("npu", node, False), "dut")
        self.assertEqual(CONTRACT.role_for_atk_task("npu", "gold", False), "benchmark")
        for node in ("gold", "npu_dut_1"):
            self.assertEqual(CONTRACT.role_for_atk_task("cpu", node, True), "golden")

    def test_dc_alias_requires_both_runtime_markers(self):
        self.assertEqual(CONTRACT.role_for_atk_task(
            "npu", "npu_dut_1", False,
            task_types=("accuracy_dc",), run_modes=(DC_MODE,)), "dut")
        for tasks, modes in (((), ()), (("accuracy",), (DC_MODE,)),
                             (("performance_device",), (DC_MODE,)), (("run",), (DC_MODE,)),
                             (("accuracy_dc",), ()), (("accuracy_dc",), ("unknown",)),
                             (("accuracy_dc", "accuracy"), (DC_MODE,))):
            with self.subTest(tasks=tasks, modes=modes), self.assertRaises(RuntimeError):
                CONTRACT.role_for_atk_task("npu", "npu_dut_1", False,
                                          task_types=tasks, run_modes=modes)

    def test_dc_does_not_expand_other_role_names(self):
        for device, node, benchmark in (("npu", "npu_dut_2", False),
                                        ("npu", "phase6_1", False),
                                        ("npu", "unknown", False),
                                        ("npu", "npu_dut_1", True),
                                        ("cpu", "npu_dut_1", False)):
            with self.subTest(device=device, node=node), self.assertRaises(RuntimeError):
                CONTRACT.role_for_atk_task(device, node, benchmark,
                                          task_types=("accuracy_dc",), run_modes=(DC_MODE,))

    def test_executor_normalizes_actual_task_enum_and_forwards_markers(self):
        tree = ast.parse((OP / "executor_chunk_gated_delta_rule_fwd.py").read_text(encoding="utf-8"))
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FunctionApi")
        init = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        init.args.args[1].annotation = None
        klass.body = [init]
        klass.decorator_list = []

        class BaseApi:
            def __init__(self, task_result):
                self.original_result = task_result

        class TaskType(Enum):
            DC = "accuracy_dc"

        env = {"BaseApi": BaseApi}
        exec(compile(ast.Module(body=[klass], type_ignores=[]), "real_executor_constructor", "exec"), env)
        result = SimpleNamespace(name="npu_dut_1", is_benchmark_task=False,
                                 case_config=SimpleNamespace(id=377),
                                 task_type=[TaskType.DC], run_modes=[DC_MODE])
        api = env["FunctionApi"](result)
        self.assertIs(api.original_result, result)
        self.assertEqual(api._task_types, ("accuracy_dc",))
        self.assertEqual(api._run_modes, (DC_MODE,))
        # 检查真实 init_by_input_data 将上述元数据传入合同，而非仅存储后未使用。
        original = ast.parse((OP / "executor_chunk_gated_delta_rule_fwd.py").read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(original) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "role_for_atk_task"]
        self.assertEqual(len(calls), 1)
        self.assertEqual({arg.arg: ast.unparse(arg.value) for arg in calls[0].keywords},
                         {"task_types": "self._task_types", "run_modes": "self._run_modes"})
        result.task_type = result.run_modes = None
        api = env["FunctionApi"](result)
        self.assertEqual((api._task_types, api._run_modes), ((), ()))


class AtkSeedEntryTest(unittest.TestCase):
    def run_entry(self, value):
        bash = os.environ.get("BASH_BIN") or shutil.which("bash")
        if not bash:
            self.skipTest("入口参数单测需要 bash；可显式设置 BASH_BIN")
        with tempfile.TemporaryDirectory(prefix="gdn_entry_") as temporary:
            root = Path(temporary)
            case = root / "cases.json"
            case.write_text(json.dumps([{"id": 0}]), encoding="utf-8")
            # 替身只收集 argv 并以 23 退出，真实入口会在报告解析前返回；不伪造 ATK Pass。
            atk = root / "fake-atk"
            atk.write_text('#!/usr/bin/env bash\nif [[ "$1" == --version ]]; then echo 26.8.8; exit 0; fi\n'
                           'printf "%s\\n" "$@" > "$GDN_TEST_ARGV"\nexit 23\n', encoding="utf-8", newline="\n")
            python = root / "python3"
            python.write_text('#!/usr/bin/env bash\nexec "$GDN_TEST_PYTHON" "$@"\n',
                              encoding="utf-8", newline="\n")
            for path in (atk, python):
                path.chmod(0o700)
            argv = root / "actual.txt"
            output = root / "result"
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(("GDN_ATK_", "ACCURACY_", "REQUIRED_ATK_VERSION", "ATK_BIN"))}
            env.update(PATH=str(root) + os.pathsep + env.get("PATH", ""),
                       ATK_BIN=atk.as_posix(), GDN_ATK_CASE_JSON=case.as_posix(),
                       GDN_ATK_RESULT_DIR=output.as_posix(), GDN_ATK_SINGLE_PROCESS="1",
                       GDN_TEST_ARGV=argv.as_posix(), GDN_TEST_PYTHON=Path(sys.executable).as_posix())
            if value is not None:
                env["GDN_ATK_DISABLE_ID_SEED"] = value
            run = subprocess.run([bash, (OP / "scripts/run_double_benchmark.sh").as_posix(), "0"],
                                 env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
            args = argv.read_text(encoding="utf-8").splitlines() if argv.exists() else []
            record = output / "command.txt"
            return run, args, record.read_text(encoding="utf-8") if record.exists() else ""

    def test_seed_switch_matches_actual_and_recorded_command(self):
        for value in (None, "0", "1"):
            with self.subTest(value=value):
                run, args, record = self.run_entry(value)
                self.assertEqual(run.returncode, 23, run.stdout + run.stderr)
                expected = int(value == "1")
                self.assertEqual(args.count("--disable_id_seed"), expected)
                self.assertEqual(record.count("--disable_id_seed"), expected)
                self.assertEqual(args.count("--single_process"), 1)

    def test_invalid_seed_switch_is_rejected_before_execution(self):
        run, args, record = self.run_entry("yes")
        self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
        self.assertEqual((args, record), ([], ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)

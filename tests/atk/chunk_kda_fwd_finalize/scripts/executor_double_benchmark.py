"""ATK 双标杆验证，保留 CPU FP64 结果。"""

from dataclasses import replace
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import torch
from atk.tasks.api_execute import register

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from executor_chunk_kda_fwd_finalize import FunctionApi, run_cpu


@register("executor_chunk_kda_fwd_finalize_dual")
class DoubleBenchmarkApi(FunctionApi):
    def __call__(self, input_data, with_output=False):
        outputs = super().__call__(input_data, with_output)
        if outputs is None or self.device not in ("npu", "pyaclnn"):
            return outputs
        import ct

        version = importlib.metadata.version("ct-tool")
        if version != "0.9.1":
            raise RuntimeError(f"CT Tool 0.9.1 required, got {version}")
        cpu_inputs = replace(self.inputs, **{
            name: getattr(self.inputs, name).cpu()
            for name in ("qg_scaled", "aqk", "v_new", "h")
        })
        golden = run_cpu(cpu_inputs, high_precision=True)
        benchmark = run_cpu(cpu_inputs, high_precision=False).bfloat16()
        dut = outputs[0].cpu()
        result = ct.dual(dut, golden, benchmark, level="L1", dtype="bfloat16")
        case_id = self.task_result.case_config.id
        root = Path(os.environ["FINALIZE_CT_OUTPUT"])
        root.mkdir(parents=True, exist_ok=True)
        record = {
            "case_id": case_id, "name": self.api_name, "ct_version": version,
            "roles": ["NPU BF16 DUT", "CPU FP64 golden", "CPU FP32/BF16 benchmark"],
            "success": bool(result["success"]),
            "ratios": {k: float(v) for k, v in result["ratios"].items()},
            "checks": {k: bool(v) for k, v in result["checks"].items()},
        }
        (root / f"case-{case_id}.json").write_text(json.dumps(record, indent=2))
        if not result["success"]:
            torch.save({"dut": dut, "golden": golden, "benchmark": benchmark,
                        "inputs": cpu_inputs}, root / f"failed-{case_id}.pt")
            raise AssertionError(f"CT dual failed for case {case_id}: {record}")
        return outputs

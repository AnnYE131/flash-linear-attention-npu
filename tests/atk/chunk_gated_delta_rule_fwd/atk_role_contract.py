#!/usr/bin/env python3
"""ATK ``cv_fused_double_benchmark`` 的三路角色合同。"""

from __future__ import annotations


RUNTIME_ROLE_ORDER = ("dut", "benchmark", "golden")
DUT_NODE_NAME = "phase6"
DUT_NODE_NAMES = (DUT_NODE_NAME, "npu_dut")
BENCHMARK_NODE_NAME = "gold"
METRIC_TENSOR_PAIRS = {
    "Actual": ("dut", "golden"),
    "Benchmark": ("benchmark", "golden"),
}


def role_for_atk_task(
    device: str,
    node_name: str,
    is_benchmark_task: bool,
    *,
    task_types: tuple[str, ...] = (),
    run_modes: tuple[str, ...] = (),
) -> str:
    """按 ATK 自动 CPU golden 与两个命名 NPU node 映射三路角色。"""

    device = str(device).strip().lower()
    node_name = str(node_name).strip()
    if device == "cpu":
        if not is_benchmark_task:
            raise RuntimeError(
                "CPU 仅允许承载 ATK 自动创建的双标杆 golden 任务："
                f"name={node_name!r} is_benchmark_task={is_benchmark_task}"
            )
        return "golden"
    if device != "npu" or is_benchmark_task:
        raise RuntimeError(
            "无法识别 ATK 双标杆任务："
            f"device={device!r} name={node_name!r} "
            f"is_benchmark_task={is_benchmark_task}"
        )
    if node_name in DUT_NODE_NAMES:
        return "dut"
    # ATK 原生 DC 的重复节点；普通 accuracy/run/performance 不开放此别名。
    if (node_name == "npu_dut_1" and tuple(task_types) == ("accuracy_dc",)
            and "ascend_use_deterministic_algorithms" in run_modes):
        return "dut"
    if node_name == BENCHMARK_NODE_NAME:
        return "benchmark"
    raise RuntimeError(
        "NPU node.name 必须为 "
        f"{DUT_NODE_NAMES!r} 中任一名称或 {BENCHMARK_NODE_NAME!r}，实际为 {node_name!r}；"
        f"DC 别名另需任务/模式校验：task_types={task_types!r} run_modes={run_modes!r}"
    )

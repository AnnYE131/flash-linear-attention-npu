"""三路 ATK 结果的布局适配；只用于结果回传后的 CPU 对比。"""

from __future__ import annotations


O_LAYOUT_BY_ROLE = {"dut": "BSND", "benchmark": "BNSD", "golden": "BNSD"}


def comparison_output_names(case, disable_recompute: bool = False):
    """返回当前输出模式的对比项；推理仍验证 o 和请求的最终状态。"""

    names = ("o", "final_state") if case.output_final_state else ("o",)
    return names if disable_recompute else names + ("g_cumsum", "A")


def select_public_outputs(outputs, case, role: str, disable_recompute: bool = False):
    """验证 DUT 可选输出合同，或从未改变的标杆结果选择对比项。"""

    if role not in O_LAYOUT_BY_ROLE:
        raise ValueError(f"无法识别输出角色：{role!r}")
    outputs = tuple(outputs)
    if role == "golden":
        expected = 4 if case.output_final_state else 3
        if len(outputs) != expected:
            raise RuntimeError(f"golden 返回项数应为 {expected}，实际为 {len(outputs)}")
        if case.output_final_state:
            o, state, g_cumsum, a = outputs
        else:
            o, g_cumsum, a = outputs
            state = None
    else:
        if len(outputs) != 4:
            raise RuntimeError(f"{role} 公开返回值必须为四元组")
        o, state, g_cumsum, a = outputs
    if o is None or (case.output_final_state and state is None):
        raise RuntimeError(f"{role} 缺少请求的 o/final_state")
    if not case.output_final_state and state is not None:
        raise RuntimeError(f"{role} 未请求 final_state 却返回了状态")
    if role == "dut" and disable_recompute:
        if g_cumsum is not None or a is not None:
            raise RuntimeError("DUT disable_recompute=True 时 g_cumsum/A 必须为 None")
    elif g_cumsum is None or a is None:
        raise RuntimeError(f"{role} 缺少 g_cumsum/A")
    result = (o, state) if case.output_final_state else (o,)
    return result if disable_recompute else result + (g_cumsum, a)


def normalize_o_for_comparison(output, case, role: str):
    """按明确的来源布局返回 BSND，不根据 shape 猜测 head/token 轴。"""

    try:
        source_layout = O_LAYOUT_BY_ROLE[role]
    except KeyError as exc:
        raise ValueError(f"无法识别输出角色：{role!r}") from exc
    public_shape = (case.batch, case.tokens, case.v_heads, case.value_dim)
    if source_layout == "BSND":
        expected_shape = public_shape
    else:
        expected_shape = (case.batch, case.v_heads, case.tokens, case.value_dim)
    actual_shape = tuple(output.shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"{role} 的 o 布局应为 {source_layout}、shape={expected_shape}，"
            f"实际 shape={actual_shape}"
        )
    if source_layout == "BNSD":
        return output.transpose(1, 2).contiguous()
    return output

"""三路 ATK 结果的布局适配；只用于结果回传后的 CPU 对比。"""

from __future__ import annotations


O_LAYOUT_BY_ROLE = {"dut": "BSND", "benchmark": "BNSD", "golden": "BNSD"}


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

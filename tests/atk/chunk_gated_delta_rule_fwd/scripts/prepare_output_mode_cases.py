#!/usr/bin/env python3
"""派生显式输出模式与种子的 ATK 矩阵，不修改冻结原始用例。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def derive_cases(cases, mode: str, seed: int):
    if mode not in ("training", "inference"):
        raise ValueError("mode 必须为 training 或 inference")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed 必须为非负整数")
    if not isinstance(cases, list) or not cases:
        raise ValueError("输入矩阵必须为非空列表")
    if [case["id"] for case in cases] != list(range(len(cases))):
        raise ValueError("冻结矩阵 id 必须按 0..N-1 连续排列")
    derived = copy.deepcopy(cases)
    for case in derived:
        if any(item["name"] == "disable_recompute" for item in case["inputs"]):
            raise ValueError("输入矩阵已包含 disable_recompute，拒绝静默覆盖")
        case["default_seed"] = seed
        case["inputs"].append({
            "name": "disable_recompute", "type": "attr", "required": True,
            "dtype": "bool", "shape": None, "range_values": mode == "inference",
            "backward": False, "align_32B": None, "outlier_values": None,
        })
    return derived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1]
                        / "atk_chunk_gated_delta_rule_fwd.json")
    parser.add_argument("--mode", required=True, choices=("training", "inference"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    cases = derive_cases(json.loads(raw), args.mode, args.seed)
    # 独占创建，不能覆盖输入或已有运行矩阵；不隐式创建目标目录。
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(cases, ensure_ascii=False) + "\n")
    print(json.dumps({
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "output_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "mode": args.mode, "seed": args.seed, "cases": len(cases),
        "output": str(args.out.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

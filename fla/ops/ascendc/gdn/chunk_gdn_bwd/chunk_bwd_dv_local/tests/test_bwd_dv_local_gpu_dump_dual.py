#!/usr/bin/env python3
"""bwd_dv_local NPU vs GPU dual benchmark using GPU-collected .pt dumps.

Compare: ct.dual(npu_out, cpu_fp64_golden, gpu_out_from_dump)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import ct
import fla_npu
import torch
import torch_npu

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

OP_NAME = "bwd_dv_local"

_BTH_TO_BHT_NAMES = frozenset({
    "q", "k", "v", "w", "u", "g", "beta", "do", "dv", "dv2", "du",
    "dq", "dk", "dw", "dg", "db", "dg2", "dk2", "v_new", "o", "A",
})
_BNTH_TO_BHNT_NAMES = frozenset({"h", "dh"})
_PASSTHROUGH = frozenset({"initial_state", "final_state", "h0", "dh0", "dht"})


def _bth_to_bht(t: torch.Tensor) -> torch.Tensor:
    if t.ndim < 3:
        return t
    return t.transpose(1, 2).contiguous()


def _bnth_to_bhnt(t: torch.Tensor) -> torch.Tensor:
    if t.ndim != 5:
        return t
    return t.permute(0, 2, 1, 3, 4).contiguous()


def _to_npu_tensor(name: str, t: Any) -> Any:
    if not isinstance(t, torch.Tensor):
        return t
    if name in _PASSTHROUGH:
        out = t
    elif name in _BNTH_TO_BHNT_NAMES:
        out = _bnth_to_bhnt(t)
    elif name in _BTH_TO_BHT_NAMES:
        out = _bth_to_bht(t)
    else:
        out = t
    return out.detach().cpu()


def _to_npu_mapping(mapping: dict[str, Any] | None) -> dict[str, Any]:
    if not mapping:
        return {}
    return {k: _to_npu_tensor(k, v) for k, v in mapping.items() if v is not None}


def _chunk_indices_npu_list(chunk_indices: Any) -> list[int] | None:
    if chunk_indices is None:
        return None
    if isinstance(chunk_indices, torch.Tensor):
        return [int(x) for x in chunk_indices.detach().cpu().reshape(-1).tolist()]
    if isinstance(chunk_indices, (list, tuple)):
        if chunk_indices and isinstance(chunk_indices[0], (list, tuple)):
            return [int(x) for pair in chunk_indices for x in pair]
        return [int(x) for x in chunk_indices]
    return None


def _prepare_chunk_indices(cu_seqlens: list[int], chunk_size: int) -> list[int]:
    out: list[int] = []
    for i in range(len(cu_seqlens) - 1):
        seq_len = int(cu_seqlens[i + 1]) - int(cu_seqlens[i])
        for chunk_idx in range((seq_len + chunk_size - 1) // chunk_size):
            out.extend([i, chunk_idx])
    return out


def load_dump_for_npu(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    meta = dict(d.get("meta") or {})
    meta["op"] = str(d["op"])
    if "inputs_npu" in d:
        inputs = d["inputs_npu"]
        outputs = d.get("outputs_npu") or {}
    else:
        inputs = _to_npu_mapping(d.get("inputs") or {})
        outputs = _to_npu_mapping(d.get("outputs") or {})
    if "chunk_indices_npu" not in meta and "chunk_indices" in meta:
        meta["chunk_indices_npu"] = meta["chunk_indices"]
    return inputs, meta, outputs


def load_case_meta(case_dir: str | Path) -> dict[str, Any]:
    path = Path(case_dir) / "case_meta.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def list_case_dirs(dump_root: str | Path) -> list[Path]:
    root = Path(dump_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dump root not found: {root}")
    return [
        p for p in sorted(root.iterdir())
        if p.is_dir() and ((p / "manifest.json").is_file() or (p / "case_meta.json").is_file())
    ]


def find_op_dump_pt(case_dir: str | Path, op_name: str) -> Path:
    case_dir = Path(case_dir)
    candidates = sorted(case_dir.glob(f"*_{op_name}.pt"))
    if not candidates:
        raise FileNotFoundError(f"no *_{op_name}.pt under {case_dir}")
    return candidates[-1]


def resolve_seq_meta(
    meta: dict[str, Any],
    case_meta: dict[str, Any],
) -> tuple[list[int] | None, list[int] | None, int, float | None]:
    cu = meta.get("cu_seqlens")
    if cu is None:
        cu = case_meta.get("cu_seqlens")
    if isinstance(cu, torch.Tensor):
        cu = [int(x) for x in cu.detach().cpu().flatten().tolist()]
    if cu is not None and len(cu) == 0:
        cu = None

    chunk_size = int(meta.get("chunk_size") or case_meta.get("chunk_size") or 64)
    chunk_indices = meta.get("chunk_indices_npu")
    if chunk_indices is None:
        chunk_indices = meta.get("chunk_indices")
    chunk_indices = _chunk_indices_npu_list(chunk_indices)
    if chunk_indices is None and cu is not None:
        chunk_indices = _prepare_chunk_indices(cu, chunk_size)

    scale = meta.get("scale")
    if scale is None:
        scale = case_meta.get("scale")
    return cu, chunk_indices, chunk_size, scale


def add_viz_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-viz", action="store_true", help="skip ct.viz after ct.dual")
    parser.add_argument(
        "-sc",
        "--sample-count",
        type=int,
        default=200_000,
        help="ct.viz ordered sample count for large tensors (default: 200000)",
    )
    parser.add_argument("--viz-dir", type=Path, default=None, help="ct.viz output directory")


def resolve_viz_dir(*, viz_dir: Path | None, pt_path: Path, default_report_dir: Path) -> Path:
    if viz_dir is not None:
        return viz_dir
    if pt_path is not None:
        return pt_path.parent / "viz"
    return default_report_dir / "viz"


def _dual_then_viz(
    tensor_name: str,
    npu_out: torch.Tensor,
    fp64_golden: torch.Tensor,
    gpu_bench: torch.Tensor,
    *,
    viz_dir: Path | str | None,
    sample_count: int = 200_000,
    enable_viz: bool = True,
    level: str = "L1",
) -> None:
    print(f"  [{tensor_name}] ct.dual(npu, cpu_fp64, gpu)", flush=True)
    result = ct.dual(npu_out.cpu(), fp64_golden, gpu_bench, level=level)
    success = result.get("success") if isinstance(result, dict) else getattr(result, "success", None)
    if success is False:
        raise AssertionError(f"{tensor_name} ct.dual failed: {result}")

    if not enable_viz or viz_dir is None:
        return
    out_dir = Path(viz_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_kwargs: dict[str, Any] = {"out_dir": str(out_dir), "name": tensor_name}
    if sample_count > 0:
        viz_kwargs["sample_count"] = int(sample_count)
    print(f"  [{tensor_name}] ct.viz(npu, cpu_fp64, sample_count={sample_count})", flush=True)
    ct.viz(npu_out.cpu(), fp64_golden, **viz_kwargs)


def compute_dv_golden_fp64(
    q: torch.Tensor,
    k: torch.Tensor,
    d_o: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float,
    cu_seqlens: list[int] | None,
    chunk_indices: list[int] | None,
    chunk_size: int,
) -> torch.Tensor:
    B, H_qk, T, K = k.shape
    H_do = d_o.shape[1]
    V = d_o.shape[-1]
    h_ratio = H_do // H_qk
    if H_do != H_qk * h_ratio:
        raise ValueError(f"H_do ({H_do}) must be a multiple of H_qk ({H_qk})")

    BT = min(chunk_size, max(16, 2 ** math.ceil(math.log2(T))))
    if cu_seqlens is not None:
        if chunk_indices is None:
            chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)
        NT = len(chunk_indices) // 2
    else:
        chunk_per_t = (T + chunk_size - 1) // chunk_size
        NT = chunk_per_t * B

    q = q.to(torch.float64)
    k = k.to(torch.float64)
    d_o = d_o.to(torch.float64)
    g = g.to(torch.float64)
    dv = torch.zeros(B, H_do, T, V, dtype=torch.float64)

    for chunk_idx in range(NT):
        if cu_seqlens is not None:
            batch_idx = int(chunk_indices[chunk_idx * 2])
            i_t = int(chunk_indices[chunk_idx * 2 + 1])
            bos = int(cu_seqlens[batch_idx])
            eos = int(cu_seqlens[batch_idx + 1])
            seq_len = eos - bos
            local_start = i_t * chunk_size
            chunk_len = min(local_start + chunk_size, seq_len) - local_start
            global_start = bos + local_start
        else:
            chunk_per_t = (T + chunk_size - 1) // chunk_size
            batch_idx = chunk_idx // chunk_per_t
            i_t = chunk_idx % chunk_per_t
            global_start = i_t * chunk_size
            chunk_len = min(global_start + chunk_size, T) - global_start
            seq_len = T
        if chunk_len <= 0:
            continue

        positions = i_t * BT + torch.arange(0, BT)
        valid = positions < seq_len
        mask = (positions.unsqueeze(1) <= positions.unsqueeze(0)) & valid.unsqueeze(1) & valid.unsqueeze(0)
        mask = mask[:chunk_len, :chunk_len]

        for qk_head in range(H_qk):
            q_chunk = q[batch_idx, qk_head, global_start:global_start + chunk_len, :]
            k_chunk = k[batch_idx, qk_head, global_start:global_start + chunk_len, :]
            attn = torch.matmul(k_chunk, q_chunk.transpose(0, 1))
            for do_group in range(h_ratio):
                do_head = qk_head * h_ratio + do_group
                g_chunk = g[batch_idx, do_head, global_start:global_start + chunk_len]
                g_factor = torch.exp(g_chunk.unsqueeze(0) - g_chunk.unsqueeze(1)) * float(scale)
                a_masked = torch.where(mask, attn * g_factor, torch.zeros_like(attn))
                do_chunk = d_o[batch_idx, do_head, global_start:global_start + chunk_len, :]
                dv[batch_idx, do_head, global_start:global_start + chunk_len, :] += torch.matmul(a_masked, do_chunk)
    return dv


def _check_op_name(pt_path: Path, meta: dict[str, Any]) -> None:
    op = str(meta.get("op") or "")
    if op and op != OP_NAME:
        raise ValueError(f"{pt_path}: expected op={OP_NAME!r}, got {op!r}")


def _resolve_scale(meta_scale: float | None, K: int) -> float:
    if meta_scale is None:
        return K ** -0.5
    return float(meta_scale)


def run_one_pt(
    pt_path: Path,
    *,
    case_meta: dict[str, Any] | None = None,
    label: str | None = None,
    verbose: bool = True,
    enable_viz: bool = True,
    sample_count: int = 200_000,
    viz_dir: Path | None = None,
) -> dict[str, Any]:
    pt_path = pt_path.resolve()
    if not pt_path.is_file():
        raise FileNotFoundError(f"dump .pt not found: {pt_path}")
    if case_meta is None:
        case_meta = load_case_meta(pt_path.parent)

    inputs, meta, gpu_outputs = load_dump_for_npu(pt_path)
    _check_op_name(pt_path, meta)

    q = inputs["q"]
    k = inputs["k"]
    d_o = inputs["do"]
    g = inputs["g"]
    gpu_dv = gpu_outputs["dv"]

    B, H_qk, T, K = k.shape
    H_do = d_o.shape[1]
    V = d_o.shape[-1]
    cu_seqlens, chunk_indices, chunk_size, meta_scale = resolve_seq_meta(meta, case_meta)
    scale = _resolve_scale(meta_scale, K)
    h_ratio = H_do // H_qk
    NT = len(chunk_indices) // 2 if chunk_indices else ((T + chunk_size - 1) // chunk_size) * B

    case_name = label or f"{pt_path.parent.name}/{pt_path.name}"
    if verbose:
        print(
            f"\n=== {case_name} ===\n"
            f"  pt: {pt_path} B={B} H_qk={H_qk} H_do={H_do} h_ratio={h_ratio} "
            f"T={T} K={K} V={V} cs={chunk_size} scale={scale:.6g} "
            f"varlen={cu_seqlens is not None} NT={NT}",
            flush=True,
        )

    t0 = time.time()
    dv_npu = torch.ops.npu.npu_chunk_bwd_dv_local(
        q.npu(),
        k.npu(),
        d_o.npu(),
        g.npu(),
        g_gamma=None,
        A=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        chunk_size=chunk_size,
    )
    torch.npu.synchronize()
    npu_elapsed = time.time() - t0

    dv_fp64 = compute_dv_golden_fp64(
        q,
        k,
        d_o,
        g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    tensor_viz_dir = None
    if enable_viz:
        base_viz_dir = resolve_viz_dir(
            viz_dir=viz_dir,
            pt_path=pt_path,
            default_report_dir=pt_path.parent,
        )
        tensor_viz_dir = base_viz_dir / case_name.replace("/", "_")

    _dual_then_viz(
        "dv",
        dv_npu,
        dv_fp64,
        gpu_dv,
        viz_dir=tensor_viz_dir,
        sample_count=sample_count,
        enable_viz=enable_viz,
    )

    return {
        "case": case_name,
        "status": "pass",
        "pt": str(pt_path),
        "npu_elapsed_s": round(npu_elapsed, 4),
        "shapes": {
            "B": B,
            "H_qk": H_qk,
            "H_do": H_do,
            "T": T,
            "K": K,
            "V": V,
            "chunk_size": chunk_size,
            "h_ratio": h_ratio,
        },
    }


def run_one_case(
    case_dir: Path,
    *,
    verbose: bool = True,
    enable_viz: bool = True,
    sample_count: int = 200_000,
    viz_dir: Path | None = None,
) -> dict[str, Any]:
    case_dir = case_dir.resolve()
    pt_path = find_op_dump_pt(case_dir, OP_NAME)
    return run_one_pt(
        pt_path,
        case_meta=load_case_meta(case_dir),
        label=case_dir.name,
        verbose=verbose,
        enable_viz=enable_viz,
        sample_count=sample_count,
        viz_dir=viz_dir or (case_dir / "viz"),
    )


def _collect_pt_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.pt is not None:
        paths.append(args.pt)
    if args.pts.strip():
        paths.extend(Path(p.strip()) for p in args.pts.split(",") if p.strip())
    return paths


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="bwd_dv_local NPU vs GPU dual benchmark from GPU dumps")
    p.add_argument("--dump-root", type=Path, default=None, help="GPU dump root for batch mode")
    p.add_argument("--pt", type=Path, default=None, help="single dump .pt file")
    p.add_argument("--pts", default="", help="comma-separated dump .pt files")
    p.add_argument("--case", default="", help="single case directory name under dump-root")
    p.add_argument("--cases", default="", help="comma-separated case names")
    p.add_argument("--phase", default="all", help="all | prefix:phase_1_ | prefix:gva_")
    p.add_argument("--report", type=Path, default=None, help="write JSON report path")
    add_viz_cli_args(p)
    return p.parse_args()


def _select_cases(dump_root: Path, args: argparse.Namespace) -> list[Path]:
    all_dirs = list_case_dirs(dump_root)
    if args.case:
        d = dump_root / args.case
        if not d.is_dir():
            raise FileNotFoundError(f"case dir not found: {d}")
        return [d]
    if args.cases.strip():
        names = [n.strip() for n in args.cases.split(",") if n.strip()]
        by_name = {p.name: p for p in all_dirs}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ValueError(f"unknown case(s): {', '.join(missing)}")
        return [by_name[n] for n in names]
    phase = args.phase.strip().lower()
    if phase in ("", "all"):
        return all_dirs
    if phase.startswith("prefix:"):
        prefix = phase.split(":", 1)[1]
        return [d for d in all_dirs if d.name.startswith(prefix)]
    raise ValueError(f"unknown --phase {args.phase!r}; use all or prefix:<name_prefix>")


def main() -> int:
    args = _parse_args()
    pt_paths = _collect_pt_paths(args)
    enable_viz = not args.no_viz
    sample_count = args.sample_count
    results: list[dict[str, Any]] = []
    failed = 0

    if pt_paths:
        for pt_path in pt_paths:
            label = pt_path.name
            try:
                results.append(run_one_pt(
                    pt_path,
                    label=label,
                    verbose=True,
                    enable_viz=enable_viz,
                    sample_count=sample_count,
                    viz_dir=args.viz_dir,
                ))
            except Exception as e:
                failed += 1
                print(f"\n=== {label} FAILED ===\n{e}", flush=True)
                traceback.print_exc()
                results.append({"case": label, "status": "fail", "pt": str(pt_path), "error": str(e)})
        default_report_dir = pt_paths[0].resolve().parent
    else:
        if args.dump_root is None:
            print("ERROR: provide --dump-root for batch mode, or --pt/--pts for a single file.", file=sys.stderr)
            return 2
        selected = _select_cases(args.dump_root, args)
        if not selected:
            print("No cases selected.", file=sys.stderr)
            return 1
        for case_dir in selected:
            try:
                results.append(run_one_case(
                    case_dir,
                    verbose=True,
                    enable_viz=enable_viz,
                    sample_count=sample_count,
                    viz_dir=args.viz_dir,
                ))
            except Exception as e:
                failed += 1
                print(f"\n=== {case_dir.name} FAILED ===\n{e}", flush=True)
                traceback.print_exc()
                results.append({"case": case_dir.name, "status": "fail", "error": str(e)})
        default_report_dir = args.dump_root

    report = {
        "op": OP_NAME,
        "mode": "pt" if pt_paths else "case_dir",
        "dump_root": str(args.dump_root) if args.dump_root else None,
        "pt_files": [str(p) for p in pt_paths] if pt_paths else None,
        "total": len(results),
        "passed": len(results) - failed,
        "failed": failed,
        "results": results,
    }
    report_path = args.report or (default_report_dir / "bwd_dv_local_gpu_dump_dual_report.json")
    with Path(report_path).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nDone: {report['passed']}/{report['total']} passed, report -> {report_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

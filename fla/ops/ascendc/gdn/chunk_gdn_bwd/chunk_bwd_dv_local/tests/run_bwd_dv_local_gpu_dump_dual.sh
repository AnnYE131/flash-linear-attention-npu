#!/usr/bin/env bash
# bwd_dv_local NPU vs GPU dual benchmark from GPU-collected dumps.
#
# Usage:
#   ./run_bwd_dv_local_gpu_dump_dual.sh [DUMP_ROOT] [extra args...]
#   ./run_bwd_dv_local_gpu_dump_dual.sh /path/to/005_bwd_dv_local.pt
#   ./run_bwd_dv_local_gpu_dump_dual.sh --pt /path/to/005_bwd_dv_local.pt
#
# Examples:
#   ./run_bwd_dv_local_gpu_dump_dual.sh /path/to/GPU_DUMP
#   ./run_bwd_dv_local_gpu_dump_dual.sh ./GPU_DUMP --case phase_1_fix_1
#   ./run_bwd_dv_local_gpu_dump_dual.sh ./GPU_DUMP/phase_1_fix_1/005_bwd_dv_local.pt -sc 100000
#   ./run_bwd_dv_local_gpu_dump_dual.sh --pt a.pt --no-viz

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEST_DEVICE_ID="${TEST_DEVICE_ID:-0}"

if [[ $# -eq 0 ]]; then
  exec python3 "${SCRIPT_DIR}/test_bwd_dv_local_gpu_dump_dual.py" --dump-root ./GPU_DUMP
fi

if [[ "${1:-}" == *.pt ]]; then
  exec python3 "${SCRIPT_DIR}/test_bwd_dv_local_gpu_dump_dual.py" --pt "$1" "${@:2}"
fi

if [[ "${1:-}" == --pt || "${1:-}" == --pts ]]; then
  exec python3 "${SCRIPT_DIR}/test_bwd_dv_local_gpu_dump_dual.py" "$@"
fi

DUMP_ROOT="${1:-./GPU_DUMP}"
shift || true

exec python3 "${SCRIPT_DIR}/test_bwd_dv_local_gpu_dump_dual.py" \
  --dump-root "${DUMP_ROOT}" \
  "$@"

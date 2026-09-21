#!/usr/bin/env bash
set -euo pipefail
op_dir=$(cd "$(dirname "$0")/.." && pwd)
device=${1:-0}
run_dir=${ATK_DOUBLE_OUTPUT:?Set ATK_DOUBLE_OUTPUT to a new results directory}
mkdir -p "$run_dir"
export FINALIZE_CT_OUTPUT="$run_dir/ct-results"
python - "$op_dir/atk_chunk_kda_fwd_finalize.json" "$run_dir/cases.json" <<'PY'
import json, sys
from pathlib import Path
cases = json.loads(Path(sys.argv[1]).read_text())
for case in cases:
    case['api_type'] = 'executor_chunk_kda_fwd_finalize_dual'
Path(sys.argv[2]).write_text(json.dumps(cases, indent=2))
PY
range_args=()
if [[ -n "${CASE_START:-}" && -n "${CASE_END:-}" ]]; then
    range_args=(-s "$CASE_START" -e "$CASE_END")
fi
cd "$run_dir"
atk node --name npu_dut --backend npu --devices "$device" \
  node --name cpu_golden --backend cpu task -c "$run_dir/cases.json" \
  -p "$op_dir/scripts/executor_double_benchmark.py" --task accuracy \
  --bm_device cpu --single_process -to 120 --save_data output "${range_args[@]}"
python - "$run_dir" "${CASE_START:-0}" "${CASE_END:-624}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
expected = set(range(int(sys.argv[2]), int(sys.argv[3])))
records = [json.loads(p.read_text()) for p in (root/'ct-results').glob('case-*.json')]
assert {r['case_id'] for r in records} == expected, 'Missing CT case results'
assert all(r['success'] for r in records), 'CT dual failure'
(root/'summary.json').write_text(json.dumps({'total':len(records),'passed':len(records),'ct_version':'0.9.1'},indent=2))
print('CT_DUAL_COMPLETE',len(records))
PY

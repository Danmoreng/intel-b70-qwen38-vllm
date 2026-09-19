#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
stamp="$(date -u +%Y%m%d-%H%M%S)"
run="$repo/benchmarks/experiments/m06-target-head-int4/runs/serving-$stamp"
resume_args=()
if [[ -n "${B70_RESUME_RUN:-}" ]]; then
  run="$B70_RESUME_RUN"
  resume_args=(--resume)
fi
candidate="${B70_CANDIDATE_IMAGE:-local/qwen38-b70-vllm:m06-target-int4-fixed-20260919}"
candidate_id="$(docker image inspect "$candidate" --format '{{.Id}}')"
unit="b70-m06-target-head-serving"
systemctl --user reset-failed "$unit.service" 2>/dev/null || true
systemd-run --user --unit="$unit" --description="B70 target-head INT4 serving gate" \
  --property=Type=exec --property=TimeoutStopSec=900 \
  --property="ExecStopPost=/usr/bin/env B70_RUN_DIR=$run python3 $repo/scripts/restore-production.py" \
  /usr/bin/python3 "$repo/scripts/run-target-head-int4-serving.py" --run-dir "$run" --candidate "$candidate_id" "${resume_args[@]}"
echo "$run"

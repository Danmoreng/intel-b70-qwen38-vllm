#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
stamp="$(date -u +%Y%m%d-%H%M%S)"
run_dir="$repo_dir/benchmarks/experiments/m03-gptq-small-m/runs/run-$stamp"
unit="b70-gptq-small-m-study"
systemctl --user reset-failed "$unit.service" 2>/dev/null || true
systemd-run --user --unit="$unit" --description="B70 M03 GPTQ small-M operator screen" \
  --property=Type=exec --property=TimeoutStopSec=900 \
  --property="ExecStopPost=/usr/bin/env B70_RUN_DIR=$run_dir python3 $repo_dir/scripts/restore-production.py" \
  /usr/bin/python3 "$repo_dir/scripts/run-gptq-small-m-study.py" --run-dir "$run_dir"
echo "$run_dir"

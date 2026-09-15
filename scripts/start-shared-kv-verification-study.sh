#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
stamp="$(date -u +%Y%m%d-%H%M%S)"
run_dir="$repo_dir/benchmarks/experiments/m04-shared-kv-verification/runs/run-$stamp"
unit="b70-shared-kv-verification-study"
systemctl --user reset-failed "$unit.service" 2>/dev/null || true
systemd-run --user --unit="$unit" --description="B70 M04 shared-KV verification attention gate" \
  --property=Type=exec --property=TimeoutStopSec=900 \
  --property="ExecStopPost=/usr/bin/env B70_RUN_DIR=$run_dir python3 $repo_dir/scripts/restore-production.py" \
  /usr/bin/python3 "$repo_dir/scripts/run-shared-kv-verification-study.py" --run-dir "$run_dir"
echo "$run_dir"

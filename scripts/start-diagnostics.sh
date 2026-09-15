#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
stamp="$(date -u +%Y%m%d-%H%M%S)"
run_dir="$repo_dir/benchmarks/experiments/m01-observability/runs/run-$stamp"
unit="b70-performance-diagnostics"

if systemctl --user is-active --quiet "$unit.service"; then
  echo "$unit.service is already running" >&2
  exit 2
fi

systemd-run --user \
  --unit="$unit" \
  --description="B70 M01 scheduler-memory and full-decode diagnostics" \
  --property=Type=exec \
  --property=TimeoutStopSec=900 \
  --property="ExecStopPost=/usr/bin/env B70_RUN_DIR=$run_dir python3 $repo_dir/scripts/restore-production.py" \
  /usr/bin/python3 "$repo_dir/scripts/run-diagnostics.py" --run-dir "$run_dir"

echo "$run_dir"
echo "status: systemctl --user status $unit.service"
echo "safe stop + production restore: systemctl --user stop $unit.service"
echo "The run is independent of the Codex task and continues while the task is paused."

#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
stamp="$(date -u +%Y%m%d-%H%M%S)"
run_dir="$repo_dir/benchmarks/experiments/m06-preemptions/runs/run-$stamp"
unit="b70-m06-preemption-study"
if systemctl --user is-active --quiet "$unit.service"; then
  echo "$unit.service is already running" >&2; exit 2
fi
systemd-run --user --unit="$unit" \
  --description="B70 sparse-retention 196K preemption qualification" \
  --property=Type=exec --property=TimeoutStopSec=900 \
  --property="ExecStopPost=/usr/bin/env B70_RUN_DIR=$run_dir python3 $repo_dir/scripts/restore-production.py" \
  /usr/bin/python3 "$repo_dir/scripts/run-preemption-study.py" --run-dir "$run_dir"
echo "$run_dir"
echo "status: systemctl --user status $unit.service"
echo "safe stop + restore: systemctl --user stop $unit.service"

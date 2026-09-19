#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:-all}"
case "$mode" in all|preflight|coding) ;; *) echo 'Usage: start.sh [all|preflight|coding]' >&2; exit 2;; esac
run="$(cat "$root/LATEST")"
python3 "$root/study.py" verify --run-dir "$run"
systemd-run --user --unit=b70-m09-runtime-coding --collect \
  --property=Type=exec --property=TimeoutStopSec=1000 \
  --property="ExecStopPost=/usr/bin/python3 $root/study.py recover --run-dir $run" \
  /usr/bin/python3 "$root/study.py" "$mode" --run-dir "$run"

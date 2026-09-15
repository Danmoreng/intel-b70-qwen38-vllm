#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
unit="b70-m05-int2-draft-head"
if systemctl --user is-active --quiet "$unit.service"; then
  echo "$unit is already running" >&2
  exit 1
fi
systemd-run --user --collect --unit "$unit" \
  --property=TimeoutStartSec=0 \
  /usr/bin/python "$repo/scripts/run-int2-draft-head-study.py"
echo "Started $unit.service"

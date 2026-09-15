#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
unit="b70-m05-hidden-capture"
systemd-run --user --collect --unit "$unit" --property=TimeoutStartSec=0 \
  /usr/bin/python "$repo/scripts/capture-draft-head-states.py"
echo "Started $unit.service"

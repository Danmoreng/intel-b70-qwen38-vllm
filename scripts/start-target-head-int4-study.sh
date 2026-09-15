#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
systemd-run --user --collect --unit b70-m06-target-head-int4 \
  --property=TimeoutStartSec=0 \
  /usr/bin/python "$repo/scripts/run-target-head-int4-study.py"

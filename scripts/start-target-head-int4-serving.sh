#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
stamp="$(date -u +%Y%m%d-%H%M%S)"
run="$repo/benchmarks/experiments/m06-target-head-int4/runs/serving-$stamp"
candidate="local/qwen38-b70-vllm:m06-target-int4-candidate"
candidate_id="$(docker image inspect "$candidate" --format '{{.Id}}')"
unit="b70-m06-target-head-serving"
systemctl --user reset-failed "$unit.service" 2>/dev/null || true
systemd-run --user --unit="$unit" --description="B70 target-head INT4 serving gate" \
  --property=Type=exec --property=TimeoutStopSec=900 \
  --property="Environment=B70_CONTROL_IMAGE=sha256:aee9857bef1f37c8f0ee136d9f89d7166201212175a8b171d958627706cf1c0b" \
  --property="Environment=B70_CANDIDATE_IMAGE=$candidate" \
  --property="Environment=B70_CANDIDATE_ID=$candidate_id" \
  --property="Environment=B70_REQUIRE_M04_DIFFERENTIAL=0" \
  --property="ExecStopPost=/usr/bin/env B70_RUN_DIR=$run python3 $repo/scripts/restore-production.py" \
  /usr/bin/python3 "$repo/scripts/run-shared-kv-verification-serving.py" --run-dir "$run"
echo "$run"

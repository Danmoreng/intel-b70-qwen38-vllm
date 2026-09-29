#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
root="$repo/benchmark-results/production-release-v1/final-regression"
fixtures="/home/sebastian/LocalLLM/intel-b70-qwen38-vllm/benchmark-results/meaningful-full-profile/run-20260923-201101-w0.00"
export B70_FROZEN_FIXTURE_ROOT="$fixtures"
mkdir -p "$root"

status() {
  python3 - "$root/status.json" "$1" "$2" <<'PY'
import json,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'state':sys.argv[2],
                                         'detail':sys.argv[3],
                                         'updated_unix':time.time()},indent=2)+'\n')
PY
}
failed() {
  status failed "${current:-preflight}"
}
trap failed ERR
status running preflight

expected_policy="$(cut -d' ' -f1 "$repo/config/production_policy.sha256")"
actual_policy="$(sha256sum "$repo/config/production_policy.json" | cut -d' ' -f1)"
[[ "$expected_policy" == "$actual_policy" ]]
for attempt in $(seq 1 180); do
  if curl --silent --fail --max-time 3 http://127.0.0.1:8081/health >/dev/null; then
    break
  fi
  sleep 5
done
curl --silent --fail --max-time 3 http://127.0.0.1:8081/health >/dev/null
image_id="$(docker inspect b70-qwen38-vllm --format '{{.Image}}')"
label="$(docker inspect b70-qwen38-vllm --format '{{index .Config.Labels "org.local.b70.policy.sha256"}}')"
[[ "$label" == "$expected_policy" ]]
printf '%s\n' "$image_id" > "$root/image-id.txt"

python3 - "$repo/config/frozen_fixture_manifest.json" "$fixtures" <<'PY'
import hashlib,json,sys
from pathlib import Path
manifest=json.loads(Path(sys.argv[1]).read_text())
root=Path(sys.argv[2])
for name,expected in manifest['files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
        raise SystemExit(f'fixture hash mismatch: {name}')
print(f"verified {len(manifest['files'])} frozen fixture files",flush=True)
PY

for task_set in 32k 128k 199k; do
  current="$task_set"
  status running "$task_set"
  echo "Starting $task_set on $image_id" >&2
  python3 "$repo/benchmarks/experiments/onednn-prefill/run_performance_tasks.py" \
    --arm production-v1 --task-set "$task_set" \
    --output "$root/$task_set.jsonl" > "$root/$task_set.log" 2>&1
  [[ "$(docker inspect b70-qwen38-vllm --format '{{.Image}}')" == "$image_id" ]]
done

python3 - "$root" "$image_id" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); image=sys.argv[2]
counts={'32k':30,'128k':12,'199k':6}
summary={'image_id':image,'task_sets':{}}
for name,expected in counts.items():
    rows=[json.loads(line) for line in (root/f'{name}.jsonl').read_text().splitlines()]
    if len(rows)!=expected or len({row['id'] for row in rows})!=expected:
        raise SystemExit(f'{name}: expected {expected} unique tasks, got {len(rows)}')
    if any(row['image']!=image for row in rows):
        raise SystemExit(f'{name}: image identity changed')
    if any(row['metrics']['preemptions'] for row in rows):
        raise SystemExit(f'{name}: preemption occurred')
    summary['task_sets'][name]={'passed':sum(row['score']['passed'] for row in rows),
                                'count':len(rows),
                                'wall_sum_s':sum(row['wall_s'] for row in rows),
                                'failures':[row['id'] for row in rows
                                            if not row['score']['passed']]}
review=next(row for row in rows if row['id']=='context-199k-review')
summary['known_199k_page_review_passed']=review['score']['passed']
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
PY
status complete regression
trap - ERR

#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
readarray -t release < <(python3 - "$repo/config/production_image.json" <<'PY'
import json, sys
from pathlib import Path
release = json.loads(Path(sys.argv[1]).read_text())
for key in ('base_image_id', 'image_tag', 'image_id'):
    print(release[key])
PY
)
[[ "${#release[@]}" == 3 ]]
[[ "$(docker image inspect "${release[0]}" --format '{{.Id}}')" == "${release[0]}" ]]
image_file="$(mktemp)"
trap 'rm -f "$image_file"' EXIT
docker build --pull=false --iidfile "$image_file" --build-arg "B70_ROW_DISPATCH_BASE=${release[0]}" \
  -f "$repo/docker/w4a8-current/Dockerfile.row-dispatch" \
  "$repo"
[[ "$(cat "$image_file")" == "${release[2]}" ]] || {
  echo "Rebuilt image differs from the validated release; review and validate before changing production_image.json" >&2
  exit 1
}
docker run --rm --entrypoint python "${release[2]}" /opt/b70/verify_production.py --files-only
docker tag "${release[2]}" "${release[1]}"

#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$#" == 0 ]] || { echo "Production launcher accepts no vLLM overrides" >&2; exit 2; }
if [[ -f "$repo_dir/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$repo_dir/.env"
  set +a
fi
exec python3 "$repo_dir/scripts/run-server-exl3.py"

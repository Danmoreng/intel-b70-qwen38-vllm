#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$#" == 0 || ( "$#" == 1 && "$1" == --check-only ) ]] || { echo 'Rollback accepts only --check-only' >&2; exit 2; }
if [[ -f "$repo_dir/.env" ]]; then
  set -a
  source "$repo_dir/.env"
  set +a
fi
export VLLM_IMAGE=local/b70-qwen38-vllm:production-exl3-v1
unset EXL3_SDPA_CACHE_CAPACITY
exec python3 "$repo_dir/scripts/run-server-exl3.py" --release-dir "$repo_dir/config/releases/exl3-v1" "$@"

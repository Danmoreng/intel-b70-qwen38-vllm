#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="${VLLM_IMAGE:-local/b70-qwen38-vllm:q128-196k-180w}"

docker build --pull=false \
  -t "$image" \
  -f "$repo_dir/docker/Dockerfile" \
  "$repo_dir/docker"

docker image inspect "$image" --format 'built {{.Id}}'

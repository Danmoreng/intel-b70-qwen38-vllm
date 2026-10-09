#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="${B70_BASE_BUILD_IMAGE:-local/b70-qwen38-vllm:base-vllm-0.31.0-xpu-kernels-0.1.15.4}"
# Validate before any download, directory creation, build or tag mutation.
python3 "$repo_dir/scripts/release_integrity.py" --check-build-tag "$image"
# The immutable official XPU base includes the qualified kernel wheel.
# Native EXL3 and M04 must be rebuilt with Torch 2.14 / C++20 separately.

docker build --pull=false \
  -t "$image" \
  -f "$repo_dir/docker/Dockerfile" \
  "$repo_dir/docker"

docker image inspect "$image" --format 'built {{.Id}}'

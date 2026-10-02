#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="${B70_BASE_BUILD_IMAGE:-local/b70-qwen38-vllm:base-vllm-0.30.0-xpu-kernels-0.1.15.4}"
# Validate before any download, directory creation, build or tag mutation.
python3 "$repo_dir/scripts/release_integrity.py" --check-build-tag "$image"
wheel_name="vllm_xpu_kernels-0.1.15.4-cp38-abi3-manylinux_2_28_x86_64.whl"
wheel_sha="4ec262f7afdd07c62defc8286dc9f357569627b996c09b410f7972b653696fe7"
wheel_dir="$repo_dir/docker/wheel"
wheel_path="$wheel_dir/$wheel_name"
mkdir -p "$wheel_dir"
if [[ ! -f "$wheel_path" ]] || ! echo "$wheel_sha  $wheel_path" | sha256sum -c - >/dev/null 2>&1; then
  curl -4 --fail --location --retry 3 --retry-all-errors \
    --output "$wheel_path.tmp" \
    "https://files.pythonhosted.org/packages/cc/34/8e09626d98f0be8e616c81884b859c1731bc815e07eb97a71c9273253264/$wheel_name"
  echo "$wheel_sha  $wheel_path.tmp" | sha256sum -c -
  mv "$wheel_path.tmp" "$wheel_path"
fi

docker build --pull=false \
  -t "$image" \
  -f "$repo_dir/docker/Dockerfile" \
  "$repo_dir/docker"

docker image inspect "$image" --format 'built {{.Id}}'

#!/usr/bin/env bash
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$here/../.." && pwd)"
expected_base="sha256:47c9d5fc5eb28fd12ef6219fbbfe7cf3863e00aa0535da377cadfc9a181bcd41"
base="$(docker image inspect local/b70-qwen38-vllm:onednn-poc-20260929 --format '{{.Id}}')"
[[ "$base" == "$expected_base" ]] || { echo "oneDNN base image changed: $base" >&2; exit 1; }

tag="${B70_MIXED_PROBE_IMAGE:-local/b70-qwen38-vllm:onednn-mixed-validate-20260929}"
docker build --pull=false -f "$here/Dockerfile.mixed-probe" -t "$tag" "$repo/docker"
docker image inspect "$tag" --format '{{.Id}}'

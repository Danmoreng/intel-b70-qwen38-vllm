#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$here/../../.." && pwd)"
kernel_source=/home/sebastian/LocalLLM/Local-AI-B70/qwen38/optimization/native-prefill-d-180w/kernel-source
sycl_tla=/home/sebastian/LocalLLM/Local-AI-B70/qwen38/optimization/attention-tiles-d-180w/sycl-tla
compiler_20260=/home/sebastian/LocalLLM/Local-AI-B70/deps/opt/intel/oneapi/compiler/2026.0
builder=local/b70-oneapi-2026.1.1-builder:vllm030
output="$here/runs/build-$(date +%Y%m%d-%H%M%S)"

test "$(docker image inspect local/qwen38-b70-vllm:vllm-0.30.0-20260923 --format '{{.Id}}')" = \
  sha256:cd6562f03c8328fe60ca69269d0e4175a284859e56525950fbfc021be19d73f3
test "$(git -C "$kernel_source" rev-parse HEAD)" = 6d92b1bfbf32767ecda8e819613eb151e70030ad
test "$(git -C "$sycl_tla" rev-parse HEAD)" = 87f6850680a580654b9ea2c80dbc01aeb36ad231
echo 'b21a4d4cdd490dcbfb892a9cf30ba260dfe9ea32e8d5c0104b62703b674a0563  '"$kernel_source/csrc/xpu/attn/xe_2/chunk_prefill.hpp" | sha256sum -c -
echo 'c2731c6bfe63bd900d69fdc300ab9c23383ee5b37c8eedd141d9c5b3cc156b3d  '"$repo/docker/q128/tiles.cpp" | sha256sum -c -
mkdir -p "$output"

for version in 2026.0 2026.1; do
  docker run --rm --entrypoint /bin/bash \
    -v "$kernel_source:/kernels:ro" \
    -v "$sycl_tla:/sycl-tla:ro" \
    -v "$repo/docker/q128/tiles.cpp:/source/tiles.cpp:ro" \
    -v "$repo/b70_ops/csrc/shared_kv_verification.cpp:/source/shared_kv_verification.cpp:ro" \
    -v "$repo/b70_ops/patches/shared-kv-verification.patch:/source/shared-kv-verification.patch:ro" \
    -v "$here/build-container.sh:/build-container.sh:ro" \
    -v "$output:/output" \
    -v "$compiler_20260:/opt/intel/oneapi/compiler/2026.0:ro" \
    "$builder" /build-container.sh "$version" 2>&1 | tee "$output/$version-build.log"
done
echo "$output"

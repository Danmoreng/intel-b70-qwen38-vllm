#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
kernel_source="${B70_KERNEL_SOURCE:-/home/sebastian/LocalLLM/Local-AI-B70/qwen38/optimization/native-prefill-d-180w/kernel-source}"
sycl_tla="${B70_SYCL_TLA:-/home/sebastian/LocalLLM/Local-AI-B70/qwen38/optimization/attention-tiles-d-180w/sycl-tla}"
compiler="${B70_ONEAPI_COMPILER:-/home/sebastian/LocalLLM/Local-AI-B70/deps/opt/intel/oneapi/compiler/2026.0}"
image="${B70_BUILD_IMAGE:-local/qwen38-b70-vllm:q128-196k-20260914}"
output="$here/build"
test "$(git -C "$kernel_source" rev-parse HEAD)" = 6d92b1bfbf32767ecda8e819613eb151e70030ad
test "$(git -C "$sycl_tla" rev-parse HEAD)" = 87f6850680a580654b9ea2c80dbc01aeb36ad231
echo 'b21a4d4cdd490dcbfb892a9cf30ba260dfe9ea32e8d5c0104b62703b674a0563  '"$kernel_source/csrc/xpu/attn/xe_2/chunk_prefill.hpp" | sha256sum -c -
test "$(docker image inspect "$image" --format '{{.Id}}')" = sha256:3f20b0cf493fe0904a7efd0ca310067790e901bc5d60203340c411e57c25010a
mkdir -p "$output"
docker run --rm --entrypoint /bin/bash \
  -v "$compiler:/opt/intel/oneapi/compiler/2026.0:ro" -v "$kernel_source:/kernels:ro" \
  -v "$sycl_tla:/sycl-tla:ro" -v "$here/csrc:/src:ro" \
  -v "$here/patches:/patches:ro" \
  -v "$here/build-container.sh:/build-container.sh:ro" -v "$output:/output" \
  "$image" /build-container.sh 2>&1 | tee "$output/build.log"
sha256sum "$output/b70_ops.so" | tee "$output/SHA256SUMS"

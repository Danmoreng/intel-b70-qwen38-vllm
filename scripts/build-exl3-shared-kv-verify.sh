#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
sources="$repo/benchmarks/experiments/exl3-shared-kv-verify"
kernel_source=/home/sebastian/LocalLLM/Local-AI-B70/qwen38/optimization/native-prefill-d-180w/kernel-source
sycl_tla=/home/sebastian/LocalLLM/Local-AI-B70/qwen38/optimization/attention-tiles-d-180w/sycl-tla
image=sha256:ae6950731b3c031f812a95c0eb615a572239426440bf67fd1b3c9c9cbfce9eca
output=${1:?Supply a new output directory}
test ! -e "$output"
test "$(git -C "$kernel_source" rev-parse HEAD)" = 6d92b1bfbf32767ecda8e819613eb151e70030ad
test "$(git -C "$sycl_tla" rev-parse HEAD)" = 87f6850680a580654b9ea2c80dbc01aeb36ad231
test -z "$(git -C "$kernel_source" status --porcelain --untracked-files=no)"
test -z "$(git -C "$sycl_tla" status --porcelain --untracked-files=no)"
mkdir -p "$output"
output=$(realpath "$output")
docker run --rm --entrypoint /bin/bash -v "$kernel_source:/kernels:ro" \
 -v "$sycl_tla:/sycl-tla:ro" -v "$sources:/src:ro" -v "$output:/output" \
 "$image" /src/build-container.sh 2>&1 | tee "$output/controller.log"

#!/usr/bin/env bash
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$here/../.." && pwd)"
workspace="$(cd -- "$repo/.." && pwd)"
dnnl="${B70_ONEDNN_INSTALL:-$workspace/b70-onednn-3.13-install}"
tag="${B70_ONEDNN_IMAGE:-local/b70-qwen38-vllm:onednn-poc-20260929}"
expected_base="sha256:f0d7bc4ea6040cf4dc8d139b01e0fb348cad80dfaecdaf8d56cd560fb9103559"
actual_base="$(docker image inspect local/b70-qwen38-vllm:w4a8-current-20260929 --format '{{.Id}}')"
[[ "$actual_base" == "$expected_base" ]] || { echo "base image changed: $actual_base" >&2; exit 1; }

"$repo/benchmarks/experiments/onednn-prefill/build_native.sh"
mkdir -p "$here/stage"
rm -f "$here/stage/native_sdpa.so" "$here/stage/libdnnl.so.3.13"
cp "$repo/benchmarks/experiments/onednn-prefill/native_sdpa.so" "$here/stage/native_sdpa.so"
cp -L "$dnnl/lib/libdnnl.so.3.13" "$here/stage/libdnnl.so.3.13"
echo "ee8fc42adcf851ea0f8013ebda24f07629f49ca6ed815835a16da45ad718ca20  $here/stage/libdnnl.so.3.13" | sha256sum -c -

docker build --pull=false -f "$here/Dockerfile" -t "$tag" "$repo/docker"
docker image inspect "$tag" --format '{{.Id}}'

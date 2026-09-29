#!/usr/bin/env bash
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd -- "$here/../../../.." && pwd)"
dnnl="${B70_ONEDNN_INSTALL:-$workspace/b70-onednn-3.13-install}"
builder="${B70_ONEDNN_BUILDER:-local/b70-oneapi-2026.1.1-builder:vllm030}"
source_name="${B70_ONEDNN_SOURCE:-native_sdpa.cpp}"
output_name="${B70_ONEDNN_OUTPUT:-native_sdpa.so}"
[[ "$source_name" =~ ^[A-Za-z0-9_.-]+\.cpp$ && -f "$here/$source_name" ]] || { echo "invalid native source: $source_name" >&2; exit 2; }
[[ "$output_name" =~ ^[A-Za-z0-9_.-]+\.so$ ]] || { echo "invalid native output: $output_name" >&2; exit 2; }
[[ -f "$dnnl/lib/libdnnl.so.3.13" ]] || { echo "missing oneDNN library: $dnnl" >&2; exit 2; }
rm -f "$here/$output_name" "$here/$output_name.tmp"

docker run --rm \
  -v "$here:/work" -v "$dnnl:/dnnl:ro" -w /work \
  -e B70_ONEDNN_SOURCE="$source_name" -e B70_ONEDNN_OUTPUT="$output_name" \
  --entrypoint /bin/bash "$builder" -lc '
set -euo pipefail
torch_root=/opt/venv/lib/python3.12/site-packages/torch
/opt/intel/oneapi/compiler/2026.1/bin/icpx \
  -fsycl -fsycl-targets=spir64 -O3 -fPIC -std=c++17 -shared \
  -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I/dnnl/include -I"$torch_root/include" \
  -I"$torch_root/include/torch/csrc/api/include" \
  "$B70_ONEDNN_SOURCE" -o "$B70_ONEDNN_OUTPUT.tmp" \
  -L"$torch_root/lib" -Wl,-rpath,"$torch_root/lib" \
  -Wl,-rpath,/dnnl/lib -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu \
  -L/dnnl/lib -ldnnl
mv "$B70_ONEDNN_OUTPUT.tmp" "$B70_ONEDNN_OUTPUT"
sha256sum "$B70_ONEDNN_OUTPUT"
'
[[ -s "$here/$output_name" ]] || { echo "native library absent after build" >&2; exit 1; }

#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
compiler=/home/sebastian/LocalLLM/Local-AI-B70/deps/opt/intel/oneapi/compiler/2026.0
image=sha256:b675d81d4e7cc63fbcd6df395965ea16ec5c4704428c81118a1618185245dd5a
mkdir -p "$root/runs/build"
docker run --rm --network=none --entrypoint bash \
  -v "$compiler:/compiler:ro" -v "$root:/src:ro" -v "$root/runs/build:/out" "$image" -c '
  export LD_LIBRARY_PATH=/compiler/lib:${LD_LIBRARY_PATH:-}
  t=/opt/venv/lib/python3.12/site-packages/torch
  /compiler/bin/icpx -fsycl -O3 -DNDEBUG -std=c++20 -fPIC -shared \
    -fno-sycl-instrument-device-code -D_GLIBCXX_USE_CXX11_ABI=1 \
    -I"$t/include" -I"$t/include/torch/csrc/api/include" \
    /src/gdn_projection.cpp -L"$t/lib" -Wl,-rpath,"$t/lib" \
    -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu -o /out/gdn_projection.so'

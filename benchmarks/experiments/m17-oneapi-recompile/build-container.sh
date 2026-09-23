#!/usr/bin/env bash
set -euo pipefail
compiler_version="$1"
output="/output/$compiler_version"
mkdir -p "$output"
export PATH="/opt/intel/oneapi/compiler/$compiler_version/bin:$PATH"
export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/$compiler_version/lib:${LD_LIBRARY_PATH:-}"
icpx --version | tee "$output/compiler-version.txt"

torch_root=/opt/venv/lib/python3.12/site-packages/torch
cp -a /kernels /tmp/b70-kernels
kernel_root=/tmp/b70-kernels
xe2="$kernel_root/csrc/xpu/attn/xe_2"
flags=(
  -fsycl -O3 -DNDEBUG -Xspirv-translator
  -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate
  -std=c++20 -fPIC -shared -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET
  -DVLLM_XPU_ENABLE_XE2 -fno-sycl-instrument-device-code
  -D_GLIBCXX_USE_CXX11_ABI=1
  "-I$torch_root/include" "-I$torch_root/include/torch/csrc/api/include"
  -I/usr/include/python3.12 "-I$kernel_root" "-I$kernel_root/csrc" "-I$xe2"
  -I/sycl-tla/include -I/sycl-tla/tools/util/include -I/sycl-tla/applications
)
links=("-L$torch_root/lib" "-Wl,-rpath,$torch_root/lib" \
  -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu)

icpx "${flags[@]}" /source/tiles.cpp "${links[@]}" -o "$output/tiles.so"
patch -d /tmp/b70-kernels -p1 < /source/shared-kv-verification.patch
icpx "${flags[@]}" /source/shared_kv_verification.cpp "${links[@]}" -o "$output/m04.so"
sha256sum "$output/tiles.so" "$output/m04.so" | tee "$output/SHA256SUMS"
ls -lh "$output/tiles.so" "$output/m04.so"

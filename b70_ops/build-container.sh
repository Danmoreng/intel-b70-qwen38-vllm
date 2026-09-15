#!/usr/bin/env bash
set -euo pipefail
export PATH=/opt/intel/oneapi/compiler/2026.0/bin:$PATH
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2026.0/lib:${LD_LIBRARY_PATH:-}
t=/opt/venv/lib/python3.12/site-packages/torch
kernel_tree=/tmp/b70-kernels
cp -a /kernels "$kernel_tree"
patch -d "$kernel_tree" -p1 < /patches/shared-kv-verification.patch
k="$kernel_tree/csrc/xpu/attn/xe_2"
icpx --version
icpx -fsycl -O3 -DNDEBUG -Xspirv-translator \
  -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
  -std=c++20 -fPIC -shared -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
  -DVLLM_XPU_ENABLE_XE2 -fno-sycl-instrument-device-code -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I"$t/include" -I"$t/include/torch/csrc/api/include" -I/usr/include/python3.12 \
  -I"$kernel_tree" -I"$kernel_tree/csrc" -I"$k" -I/sycl-tla/include \
  -I/sycl-tla/tools/util/include -I/sycl-tla/applications \
  /src/q128_attention.cpp /src/gptq_small_m.cpp /src/shared_kv_verification.cpp \
  -L"$t/lib" -Wl,-rpath,"$t/lib" \
  -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu -o /output/b70_ops.so

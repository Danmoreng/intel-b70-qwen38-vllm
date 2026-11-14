#!/usr/bin/env bash
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u
mkdir /native
icpx --version > /native/compiler-version.txt
# Retain the reference CPU-NONE/GPU-SYCL/Graph feature set. Bound compilation
# to four jobs; the host wrapper also caps container memory and total runtime.
cmake -S /sources/onednn -B /build-onednn -G Ninja \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
 -DCMAKE_INSTALL_PREFIX=/opt/b70 -DONEDNN_CPU_RUNTIME=NONE \
 -DONEDNN_GPU_RUNTIME=SYCL -DONEDNN_GPU_VENDOR=INTEL \
 -DONEDNN_BUILD_GRAPH=ON -DONEDNN_BUILD_TESTS=OFF -DONEDNN_BUILD_EXAMPLES=OFF
cmake --build /build-onednn --parallel 4
cmake --install /build-onednn
cp -L /opt/b70/lib/libdnnl.so.3 /opt/b70/libdnnl.so.3
cd /matrix-source
EXL3_OUT=/native/_C.so EXL3_BUILD_LOG=/native/exl3-compiler.log \
 EXL3_DNNL_ROOT=/opt/b70 EXL3_FLAGS=-DEXL3_DPAS_MB4 bash scripts/build_ext.sh
# M04 packed verification, rebased onto matching current XE2 header inputs.
cp -a /sources/xpu-kernels /m04-kernels
patch --batch --fuzz=0 -d /m04-kernels -p1 < /m04-source/patches/shared-kv-verification.patch
T=$(python -c 'import torch,pathlib; print(pathlib.Path(torch.__file__).parent)')
ABI=$(python -c 'import torch; assert torch.__version__ == "2.15.0+xpu"; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
command=(icpx -fsycl -O3 -DNDEBUG -Xspirv-translator
 -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate
 -std=c++20 -fPIC -shared -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET
 -DVLLM_XPU_ENABLE_XE2 -fno-sycl-instrument-device-code "-D_GLIBCXX_USE_CXX11_ABI=$ABI"
 "-I$T/include" "-I$T/include/torch/csrc/api/include" -I/usr/include/python3.12
 -I/m04-kernels -I/m04-kernels/csrc -I/m04-kernels/csrc/xpu/attn/xe_2
 -I/sources/sycl-tla/include -I/sources/sycl-tla/tools/util/include -I/sources/sycl-tla/applications
 /m04-source/csrc/shared_kv_verification.cpp "-L$T/lib" "-Wl,-rpath,$T/lib"
 -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu -o /native/m04.so)
printf '%s\n' "${command[@]}" > /native/m04-compiler-argv.txt
"${command[@]}" 2>&1 | tee /native/m04-compiler.log
python /build-tools/write-m04-receipt.py
# Preserve dependency and toolchain identities alongside the output binaries.
dpkg-query -W > /native/dpkg-packages.txt
python -m pip freeze > /native/python-packages.txt

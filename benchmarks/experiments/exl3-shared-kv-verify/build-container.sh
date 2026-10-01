#!/usr/bin/env bash
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u
test ! -e /output/m04.so
kernel_tree=/tmp/b70-exl3-m04-kernels
cp -a /kernels "$kernel_tree"
patch --batch --fuzz=0 -d "$kernel_tree" -p1 < /src/patches/shared-kv-verification.patch
torch_root=$(python3 -c 'import torch,pathlib; print(pathlib.Path(torch.__file__).parent)')
abi=$(python3 -c 'import torch; assert torch.__version__ == "2.13.0+xpu"; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
temporary=$(mktemp /output/m04.pending.XXXXXX.so)
trap 'rm -f "$temporary"' EXIT
kernel_include="$kernel_tree/csrc/xpu/attn/xe_2"
command=(icpx -fsycl -O3 -DNDEBUG -Xspirv-translator
 -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate
 -std=c++20 -fPIC -shared -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET
 -DVLLM_XPU_ENABLE_XE2 -fno-sycl-instrument-device-code "-D_GLIBCXX_USE_CXX11_ABI=$abi"
 "-I$torch_root/include" "-I$torch_root/include/torch/csrc/api/include" -I/usr/include/python3.12
 "-I$kernel_tree" "-I$kernel_tree/csrc" "-I$kernel_include" -I/sycl-tla/include
 -I/sycl-tla/tools/util/include -I/sycl-tla/applications
 /src/csrc/shared_kv_verification.cpp "-L$torch_root/lib" "-Wl,-rpath,$torch_root/lib"
 -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu -o "$temporary")
printf '%s\n' "${command[@]}" > /output/compiler-argv.txt
icpx --version > /output/compiler-version.txt
"${command[@]}" 2>&1 | tee /output/build.log
M04_PENDING_LIBRARY="$temporary" python3 - <<'PY'
import hashlib,json,os,pathlib,subprocess,torch
p=pathlib.Path(os.environ['M04_PENDING_LIBRARY'])
torch.ops.load_library(str(p))
name='b70_exl3_attention::shared_kv_verify_out'
assert torch._C._dispatch_has_kernel_for_dispatch_key(name,'XPU')
out=pathlib.Path('/output')
sources={str(x.relative_to('/src')):hashlib.sha256(x.read_bytes()).hexdigest() for x in pathlib.Path('/src').rglob('*') if x.is_file() and '__pycache__' not in str(x)}
d={'status':'COMPILE_AND_CPU_IMPORT_PASS_XPU_UNTESTED','torch':torch.__version__,'cxx11_abi':torch._C._GLIBCXX_USE_CXX11_ABI,'library_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_sha256':sources,'operator':name,'ldd':subprocess.check_output(['ldd',str(p)],text=True),'compiler_version':(out/'compiler-version.txt').read_text(),'compiler_argv':(out/'compiler-argv.txt').read_text().splitlines()}
assert 'not found' not in d['ldd']
(out/'manifest.json').write_text(json.dumps(d,indent=2)+'\n')
PY
chmod 644 "$temporary"
mv "$temporary" /output/m04.so

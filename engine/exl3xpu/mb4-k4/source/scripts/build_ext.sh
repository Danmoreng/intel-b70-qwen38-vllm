#!/usr/bin/env bash
# A stale installed library can never turn a failed compile/import into success.
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u
cd "$(dirname "$0")/.."
out=${EXL3_OUT:-exl3xpu/_C.so}
mkdir -p "$(dirname "$out")"
out=$(realpath -m "$out")
tmp=$(mktemp "${out}.pending.XXXXXX.so")
manifest_tmp=$(mktemp "${out}.manifest.pending.XXXXXX")
flags_tmp=$(mktemp "${out}.flags.pending.XXXXXX")
trap 'rm -f "$tmp" "$manifest_tmp" "$flags_tmp"' EXIT
log=${EXL3_BUILD_LOG:-${out}.build-$(date -u +%Y%m%dT%H%M%S)-$$.log}
T=$(python3 -c 'import torch,os; print(os.path.dirname(torch.__file__))')
abi=$(python3 -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
D=${EXL3_DNNL_ROOT:-/opt/intel/oneapi/dnnl/latest}
dnnl=0
dnnl_flags=(); dnnl_libs=()
if [[ -f "$D/include/oneapi/dnnl/dnnl_sycl.hpp" && -z "${EXL3_NO_DNNL:-}" ]]; then
  dnnl=1
  dnnl_flags=(-DEXL3_DNNL "-I$D/include")
  dnnl_libs=("-L$D/lib" "-Wl,-rpath,$D/lib" -ldnnl)
fi
# Parse optional flags as words; do not execute shell expansions in them.
python3 -c 'import os,shlex,sys; sys.stdout.write("\0".join(shlex.split(os.environ.get("EXL3_FLAGS","")))+"\0")' > "$flags_tmp"
mapfile -d '' -t extra_flags < "$flags_tmp"
if [[ ${#extra_flags[@]} == 1 && -z ${extra_flags[0]} ]]; then extra_flags=(); fi
compiler=${EXL3_CXX:-icpx}
command=("$compiler" -fsycl -fsycl-targets=spir64 -O3 -ffast-math -fPIC -std=c++20 -shared
 -fsycl-device-code-split=per_kernel "-D_GLIBCXX_USE_CXX11_ABI=$abi" "${extra_flags[@]}" "${dnnl_flags[@]}"
 -I csrc "-I$T/include" "-I$T/include/torch/csrc/api/include"
 -x c++ csrc/exl3_ops.sycl -x none -o "$tmp"
 "-L$T/lib" "-Wl,-rpath,$T/lib" -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu "${dnnl_libs[@]}")
# pipefail preserves the compiler status even when tee succeeds.
"${command[@]}" 2>&1 | tee "$log"
EXL3_DNNL_ROOT="$D" python3 scripts/native_build_manifest.py --library "$tmp" --output "$manifest_tmp" \
  --installed-output "$out" --log "$log" --dnnl "$dnnl" --compiler "$compiler" -- "${command[@]:1}"
chmod 644 "$tmp" "$manifest_tmp"
mv -f "$tmp" "$out"
mv -f "$manifest_tmp" "${out}.manifest.json"
printf 'Installed verified library: %s\nManifest: %s\nBuild log: %s\n' "$out" "${out}.manifest.json" "$log"

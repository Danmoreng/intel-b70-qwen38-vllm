#!/usr/bin/env python3
"""Load a candidate native library in this Torch ABI, validate registrations, attest it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import torch

XPU_OPS = ('linear','exl3_gemm_small','exl3_had_in_rm','exl3_had_out_h',
           'exl3_reconstruct','exl3_gemm_raw','exl3_fa_fwd')
HOST_OPS = ('exl3_supported','exl3_set_int8','exl3_set_target_threads',
            'exl3_set_target_threads_mb64','exl3_set_target_threads_mb16',
            'exl3_set_vec_max_m','exl3_set_fused','exl3_set_local','exl3_set_max_mb')

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('library','output','installed-output','log','compiler'): p.add_argument('--'+key,required=True)
    p.add_argument('--dnnl',type=int,choices=(0,1),required=True)
    p.add_argument('flags',nargs=argparse.REMAINDER)
    a=p.parse_args()
    torch.ops.load_library(a.library)
    xpu=XPU_OPS+(('exl3_sdpa','exl3_sdpa_len','exl3_sdpa_cache_stats') if a.dnnl else ())
    for op in HOST_OPS+xpu:
        if not hasattr(torch.ops.exl3xpu_C,op): raise RuntimeError('Missing registered operator: '+op)
    for op in xpu:
        if not torch._C._dispatch_has_kernel_for_dispatch_key('exl3xpu_C::'+op,'XPU'):
            raise RuntimeError('Missing XPU implementation: '+op)
    for bits in (4,6):
        if not torch.ops.exl3xpu_C.exl3_supported(bits,2): raise RuntimeError(f'Missing mul1/{bits} bpw support')
    root=Path(__file__).resolve().parents[1]
    source=list((root/'csrc').rglob('*'))+[Path(__file__),root/'scripts/build_ext.sh']
    versions={}
    for name in ('intel-igc-core-2','intel-igc-opencl-2','libze-intel-gpu1','intel-opencl-icd'):
        r=subprocess.run(['dpkg-query','-W','-f=${Version}',name],capture_output=True,text=True)
        versions[name]=r.stdout if r.returncode==0 else None
    dnnl_root=Path(os.environ.get('EXL3_DNNL_ROOT','/opt/intel/oneapi/dnnl/latest'))
    headers={str(h):h.read_text() for h in (dnnl_root/'include').rglob('dnnl_version.h')}
    d={'schema':1,'build_unix':time.time(),'source_sha256':{
          str(f.relative_to(root)):sha(f) for f in sorted(source) if f.is_file()},
       'compiler':a.compiler,'compiler_version':subprocess.check_output([a.compiler,'--version'],text=True),
       'flags':a.flags[1:] if a.flags[:1]==['--'] else a.flags,
       'torch_version':torch.__version__,'torch_cxx11_abi':bool(torch._C._GLIBCXX_USE_CXX11_ABI),
       'python_version':platform.python_version(),'oneDNN_enabled':bool(a.dnnl),
       'oneDNN_version_headers':headers,'driver_packages':versions,
       'oneDNN_library_sha256':{str(f):sha(f) for f in (dnnl_root/'lib').glob('libdnnl.so.*') if f.is_file()},
       'registered_ops':list(HOST_OPS+xpu),'xpu_kernel_ops':list(xpu),
       'operator_smoke':'PASS (registration and host capability queries; GPU numerical tests separate)',
       'library':a.installed_output,'library_bytes':Path(a.library).stat().st_size,'library_sha256':sha(a.library),
       'compiler_log':a.log,'compiler_log_sha256':sha(a.log)}
    Path(a.output).write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps({'native_smoke':'PASS','ops':len(d['registered_ops']),'sha256':d['library_sha256']}))

if __name__=='__main__': main()

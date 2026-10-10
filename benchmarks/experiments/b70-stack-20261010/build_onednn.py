#!/usr/bin/env python3
"""E01: rebuild pinned stable custom oneDNN with original GPU/Graph options."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
BUILDER='sha256:a78ff8ea07752bb3bc156cd47043386777031f54531075060889f5b2b55b0505'
SOURCE=Path('/home/sebastian/LocalLLM/b70-stack-onednn-3.13.4-20261010')
COMMIT='c531fe1a5cd631b7f60277cdf6675c588cadbc28'
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True)
    (out/'build').mkdir();(out/'install').mkdir()
    old=json.loads((REPO/'benchmark-results/b70-stack-20261010/oneDNN-provenance-r2/audit.json').read_text())
    flags=['-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx','-DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx','-DCMAKE_INSTALL_PREFIX=/opt/dnnl']
    for key,value in old['cmake_options'].items():
        name,kind=key.split(':')
        if name.startswith('ONEDNN_') and kind in ('BOOL','STRING','PATH'):flags.append('-D'+name+'='+value)
    script='''set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
test -x /opt/intel/oneapi/compiler/2026.1/bin/icpx
icpx --version > /results/compiler-version.txt
cmake -S /onednn -B /build -G Ninja "$@"
cmake --build /build --parallel 8
cmake --install /build
python - <<'PY'
import ctypes,hashlib,json,subprocess
from pathlib import Path
class V(ctypes.Structure):
 _fields_=[('major',ctypes.c_int),('minor',ctypes.c_int),('patch',ctypes.c_int),('hash',ctypes.c_char_p),('cpu',ctypes.c_uint),('gpu',ctypes.c_uint)]
p=Path('/opt/dnnl/lib/libdnnl.so.3');lib=ctypes.CDLL(str(p));lib.dnnl_version.restype=ctypes.POINTER(V);v=lib.dnnl_version().contents
assert (v.major,v.minor,v.patch)==(3,13,4)
assert v.hash.decode()=='c531fe1a5cd631b7f60277cdf6675c588cadbc28'
ldd=subprocess.check_output(['ldd',str(p)],text=True);assert 'not found' not in ldd
d=dict(status='BUILD_AND_CPU_LIBRARY_LOAD_PASS_GPU_UNTESTED',version='3.13.4',git=v.hash.decode(),cpu_runtime=v.cpu,gpu_runtime=v.gpu,library_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),library_bytes=p.stat().st_size,ldd=ldd,header_hashes={str(x.relative_to('/opt/dnnl')):hashlib.sha256(x.read_bytes()).hexdigest() for x in Path('/opt/dnnl/include').rglob('*') if x.is_file()})
Path('/results/library-receipt.json').write_text(json.dumps(d,indent=2)+'\\n')
PY
'''
    (out/'build-container.sh').write_text(script)
    command=['docker','run','--rm','--name','b70-stack-e01-onednn-build','--network','none','--cpus','8','--memory','24g',
        '-v',str(SOURCE)+':/onednn:ro','-v',str(out/'build')+':/build','-v',str(out/'install')+':/opt/dnnl',
        '-v',str(out)+':/results','--entrypoint','bash',BUILDER,'/results/build-container.sh',*flags]
    receipt=dict(status='RUNNING',candidate='E01',source_commit=COMMIT,builder_image=BUILDER,command=command,
        original_build_cache_sha256=old['build_cache_sha256'],logical_change='Custom oneDNN 3.13.0 to stable 3.13.4, original CPU-NONE/GPU-SYCL/Graph build flags retained',started_unix=time.time())
    def save():(out/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save();print('BUILD_ONEDNN_3_13_4',flush=True)
    try:
        with (out/'build.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        receipt.update(status='BUILD_AND_CPU_LOAD_PASS_GPU_UNTESTED',library=json.loads((out/'library-receipt.json').read_text()))
    except BaseException as e:receipt.update(status='FAILED',error=repr(e));raise
    finally:receipt['finished_unix']=time.time();save()
    print('ONE_DNN_BUILD_CPU_PASS_GPU_UNTESTED',flush=True)
if __name__=='__main__':main()

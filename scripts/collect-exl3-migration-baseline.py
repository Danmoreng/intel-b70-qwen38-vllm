#!/usr/bin/env python3
"""Read-only migration baseline; writes evidence exclusively, never touches serving."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(args, **kw):
    return subprocess.run(args, check=True, capture_output=True, text=True, **kw).stdout.strip()

def git(path):
    files = run(['git', '-C', str(path), 'ls-files', '-z', '--cached', '--others', '--exclude-standard']).split('\0')
    return {'revision': run(['git', '-C', str(path), 'rev-parse', 'HEAD']),
            'status': run(['git', '-C', str(path), 'status', '--porcelain']),
            'files': {n: digest(path / n) for n in files if n and (path / n).is_file()}}

PROBE = r'''
import sys,os,hashlib,json,platform,subprocess,importlib.util,importlib.metadata
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
interactive=importlib.util.find_spec('vllm').origin
sys.path[0]='/opt/venv/bin'
console=importlib.util.find_spec('vllm').origin
versions={}
for n in ('torch','vllm','vllm-xpu-kernels','triton','pytorch-triton-xpu'):
    try: versions[n]=importlib.metadata.version(n)
    except importlib.metadata.PackageNotFoundError: versions[n]=None
packages={}
for n in ('intel-igc-core-2','intel-igc-opencl-2','intel-ocloc','intel-opencl-icd','libze-intel-gpu1','libigdgmm12'):
    p=subprocess.run(['dpkg-query','-W','-f=${Version}',n],capture_output=True,text=True)
    packages[n]=p.stdout if p.returncode==0 else None
workers={}; native=set()
for proc in Path('/proc').glob('[0-9]*'):
    try:
        cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if not cmd.startswith('VLLM::'): continue
        workers[proc.name]={'command':cmd,'cwd':os.readlink(proc/'cwd')}
        for row in (proc/'maps').read_text().splitlines():
            fields=row.split()
            if len(fields)>=6 and fields[-1].startswith('/') and '.so' in fields[-1]:
                native.add(fields[-1])
    except (OSError,UnicodeDecodeError): pass
native.update(str(p) for root in ('/opt/b70','/opt/exl3xpu/exl3xpu')
              for p in Path(root).glob('*.so*') if p.is_file())
sources={}
for root in ('/workspace/vllm/vllm','/opt/venv/lib/python3.12/site-packages/vllm'):
    for n in ('model_executor/models/qwen3_5_mtp.py','v1/worker/xpu_worker.py',
              'v1/worker/xpu_model_runner.py','v1/worker/gpu_model_runner.py',
              'v1/worker/gpu_model_runner_v2.py','v1/core/sched/scheduler.py',
              'v1/core/kv_cache_manager.py','v1/core/kv_cache_utils.py'):
        p=Path(root)/n
        if p.is_file(): sources[str(p)]=sha(p)
for p in Path('/opt/venv/lib/python3.12/site-packages').glob('b70*.py'):
    sources[str(p)]=sha(p)
compiler=Path('/opt/intel/oneapi/compiler/latest/bin/icpx')
cv=subprocess.run([str(compiler),'--version'],capture_output=True,text=True).stdout if compiler.exists() else None
dnnl={str(p):sha(p) for p in Path('/opt/intel/oneapi/dnnl/latest/include').rglob('dnnl_version.h')}
print(json.dumps({'python':platform.python_version(),'packages':versions,'driver_packages':packages,
 'interactive_vllm_origin':interactive,'console_vllm_origin':console,'console_sys_path':sys.path,
 'source_hashes':sources,'workers':workers,'native_libraries':{n:sha(n) for n in sorted(native) if Path(n).is_file()},
 'compiler_version':cv,'dnnl_version_headers':dnnl}))
'''

def probe(container=None, image=None):
    command = ['docker', 'exec', '-i', container, 'python', '-'] if container else [
        'docker', 'run', '--rm', '-i', '--entrypoint', 'python', image, '-']
    return json.loads(run(command, input=PROBE))

def model(path, revision):
    idx = json.loads((path / 'model.safetensors.index.json').read_text())
    names = set(idx['weight_map'].values()) | {'model.safetensors.index.json','config.json'}
    names.update(p.name for p in path.glob('*.json') if p.is_file())
    return {'path':str(path),'revision':revision,'files':{
        n:{'bytes':(path/n).stat().st_size,'sha256':digest(path/n)} for n in sorted(names)}}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=REPO/'benchmarks/results/exl3-migration/baseline_environment.json')
    a=ap.parse_args()
    if a.output.exists(): raise SystemExit('Baseline is immutable; choose another output path')
    gptq=ROOT/'intel-b70-qwen38-vllm-onednn'; exl=ROOT/'exl3xpu'
    inspect=json.loads(run(['docker','inspect','b70-qwen38-vllm']))[0]
    exlimage=json.loads(run(['docker','image','inspect','local/exl3xpu:15ded2f']))[0]
    env={n:v for s in inspect['Config']['Env'] for n,_,v in [s.partition('=')]
         if n.startswith(('B70_','EXL3_','VLLM_','ZE_','SYCL_','ONEAPI_','PYTORCH_')) or n in ('PYTHONPATH','HF_HUB_OFFLINE')}
    result={'schema':1,'captured_unix':time.time(),'purpose':'immutable pre-migration evidence, not new benchmark results',
            'repositories':{'gptq':git(gptq),'exl3':git(exl)},
            'gptq':{'image':inspect['Image'],'tag':inspect['Config']['Image'],
                    'command':[inspect['Path']]+inspect['Args'],'environment':env,
                    'mounts':inspect['Mounts'],'runtime':probe(container='b70-qwen38-vllm')},
            'exl3':{'image':exlimage['Id'],'tag':'local/exl3xpu:15ded2f',
                    'runtime':probe(image=exlimage['Id']),
                    'candidate_profile_sha256':digest(exl/'models/qwen3.8-27b-exl3-4.00bpw/model.yaml'),
                    'existing_launch_profile':(exl/'models/qwen3.8-27b-exl3-4.00bpw/model.yaml').read_text()},
            'host':{'kernel':run(['uname','-a']), 'memory':Path('/proc/meminfo').read_text(),
                    'gpu':run(['lspci','-nnk']),
                    'power_caps_uw':{str(p):int(p.read_text()) for p in Path('/sys/bus/pci/devices/0000:03:00.0/hwmon').glob('*/power1_cap')}},
            'rollback':{'service':'b70-qwen38-vllm.service','image':inspect['Image'],'tested_this_phase':False}}
    evidence={}
    sources=[gptq/'benchmarks/runs/2026-09-29-production/summary.json',
             gptq/'benchmarks/runs/2026-10-01-flappybird/comparison.json',
             gptq/'benchmarks/runs/2026-10-01-flappybird/cold-comparison.json',
             gptq/'benchmark-results/quantization-reference-20261001/comparison.json',
             gptq/'benchmark-results/quantization-reference-20261001/comparison-foem.json',
             gptq/'benchmark-results/quantization-reference-20261001/foem-assessment.json',
             gptq/'benchmark-results/quantization-reference-20261001/completed.json',
             gptq/'benchmark-results/quantization-reference-20261001/original-checkpoint.json']
    dest=a.output.parent/'baseline-evidence'; dest.mkdir(parents=True,exist_ok=True)
    for p in sources:
        rel=str(p.relative_to(gptq)); target=dest/rel.replace('/','__')
        if target.exists() and digest(target)!=digest(p): raise RuntimeError(f'Immutable evidence changed: {target}')
        if not target.exists(): shutil.copyfile(p,target)
        evidence[rel]={'archived_as':str(target.relative_to(a.output.parent)),'sha256':digest(p),'bytes':p.stat().st_size}
    result['existing_evidence']=evidence
    print('Runtime and prior evidence collected; hashing checkpoint artifacts',flush=True)
    rev='a47b0c6f0d756bc394c4cc629d5b0ded1acc7001'
    hf=Path.home()/'.cache/huggingface/hub/models--mikeinnyc--Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16/snapshots'/rev
    result['models']={'gptq':model(hf,rev), 'exl3':model(Path.home()/'.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw','113cf7ab958054860e43fb7f3063b1af19171095')}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps({'output':str(a.output),'sha256':digest(a.output),'models':list(result['models'])}))

if __name__=='__main__': main()

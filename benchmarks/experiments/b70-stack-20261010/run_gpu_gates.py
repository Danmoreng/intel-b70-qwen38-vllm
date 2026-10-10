#!/usr/bin/env python3
"""Exclusive changed-stack GPU gates; always restore the pinned serving image."""
import argparse,fcntl,importlib.util,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from exl3_candidate_worker import MODEL,profile
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
restore=module('gate_restore',REPO/'scripts/run-exl3-review-micro.py')
def main():
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--m04',type=Path);p.add_argument('--attention-only',action='store_true');p.add_argument('--m04-probe-only',action='store_true');p.add_argument('--cache-pressure-only',action='store_true');a=p.parse_args()
    assert sum([a.attention_only,a.m04_probe_only,a.cache_pressure_only])<=1
    image=subprocess.check_output(['docker','image','inspect',a.image,'--format','{{.Id}}'],text=True).strip()
    baseline=json.loads((HERE/'baseline-manifest.json').read_text())['production']['image_id']
    root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
    cfg=profile(image);r=dict(status='RUNNING',image_id=image,commands=[],started_unix=time.time())
    snapshot=root/'diagnostic-sources';snapshot.mkdir()
    for name in ['m04_probe.py','cache_pressure.py','run_gpu_gates.py']:(snapshot/name).write_bytes((HERE/name).read_bytes())
    r['diagnostic_source_sha256']={p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')}
    def save():(root/'campaign.json').write_text(json.dumps(r,indent=2)+'\n')
    env={**cfg['env'],'EXL3_SDPA_CACHE_CAPACITY':'64','HF_HUB_OFFLINE':'1','ZE_AFFINITY_MASK':'0','MODEL':'/models/checkpoint'}
    common=['docker','run','--rm','--name','b70-stack-gpu-gates','--network','none','--device','/dev/dri','--shm-size','4g',
        '-v',str(MODEL)+':/models/checkpoint:ro','-v',str(root)+':/results',
        '-v',str(REPO/'benchmarks/experiments/exl3-review-20261002')+':/review:ro',
        '-v',str(snapshot)+':/experiment:ro']
    if a.m04:common+=['-v',str(a.m04.resolve())+':/candidate-m04.so:ro']
    for k,v in env.items():common+=['-e',k+'='+v]
    def command(name,args,extra=None):
        c=common+(extra or [])+['--entrypoint','python',image,'-u',*args]
        row=dict(name=name,command=c,status='RUNNING',started_unix=time.time());r['commands'].append(row);save();print('START',name,flush=True)
        with (root/(name+'.log')).open('w') as log:subprocess.run(c,check=True,stdout=log,stderr=subprocess.STDOUT)
        row.update(status='PASS',finished_unix=time.time());save();print('PASS',name,flush=True)
    with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save()
        try:
            subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
            names=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
            assert names==['b70-chatui'],names
            command('artifact-import',['/opt/exl3xpu/scripts/verify_native_artifact.py'])
            library='/candidate-m04.so' if a.m04 else '/opt/exl3xpu/m04/m04.so'
            if a.cache_pressure_only:
                command('inflight-cache-pressure',['/experiment/cache_pressure.py','--out','/results/cache-pressure.json'])
            elif a.m04_probe_only:
                command('m04-component-probe',['/experiment/m04_probe.py','--library',library,'--out','/results/m04-probe'])
            else:
                command('attention-numerics-graphs',['/opt/exl3xpu/tests/test_guarded_attention_xpu.py','--library',library,'--out','/results/attention'])
                if a.m04:
                    command('m04-component-probe',['/experiment/m04_probe.py','--library',library,'--out','/results/m04-probe'])
            if not a.attention_only and not a.m04_probe_only and not a.cache_pressure_only:
                # Creation logging is diagnostic only; no execution profiling is
                # requested from the nonprofiling production SYCL queue.
                command('implementation-diagnostic',['-c',"import torch; from exl3xpu.ops import _get_esimd; o=_get_esimd(); q=torch.randn(6,256,256,device='xpu',dtype=torch.float16); k=torch.randn(1,4097,256,device='xpu',dtype=torch.float16); y=torch.empty_like(q); e=torch.empty(0,device='xpu'); o.exl3_sdpa(q,k,k,y,e,True,.0625); torch.xpu.synchronize(); print('IMPLEMENTATION_DIAGNOSTIC_COMPLETE')"],['-e','ONEDNN_VERBOSE=dispatch,profile_create'])
                command('int8-prefill-reference',['/opt/exl3xpu/tests/test_int8_prefill.py','4096','--report','/results/int8.json'])
                command('bounded-cache-eviction-soak',['/review/cache_soak.py','--out','/results/cache-soak.json'])
            r['status']='PASS_CHANGED_STACK_LOCAL_GPU_GATES_CAPACITY_SERVING_PENDING'
        except BaseException as e:r.update(status='FAILED',error=repr(e));raise
        finally:
            subprocess.run(['docker','rm','-f','b70-stack-gpu-gates'],capture_output=True,timeout=60)
            subprocess.run(['systemctl','--user','start','b70-qwen38-vllm.service'],check=True,timeout=90)
            r['restoration']=restore.wait_for_qualified_service(baseline,timeout=900)
            r['finished_unix']=time.time();save();print('RESTORED_QUALIFIED_PRODUCTION',flush=True)
if __name__=='__main__':main()

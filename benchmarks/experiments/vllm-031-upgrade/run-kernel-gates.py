#!/usr/bin/env python3
"""Requalify a new vLLM/Torch ABI; never invoke an old baseline runtime."""
import argparse, fcntl, hashlib, json, subprocess, sys, time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'scripts'))
from exl3_candidate_worker import profile, MODEL
p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
image=subprocess.check_output(['docker','inspect',a.image,'--format','{{.Id}}'],text=True).strip()
root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
state=dict(image_id=image,status='RUNNING',started_unix=time.time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks={})
def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
cfg=profile(image)
common=['docker','run','--rm','--name','b70-v031-kernel-gates','--network','none','--device','/dev/dri','--shm-size','4g','-v',str(MODEL)+':/models/checkpoint:ro','-v',str(root)+':/results']
for k,v in {**cfg['env'],'EXL3_LIB':'/opt/exl3xpu/exl3xpu/_C.so','EXL3_SDPA_CACHE_CAPACITY':'64','HF_HUB_OFFLINE':'1','ZE_AFFINITY_MASK':'0','OMP_NUM_THREADS':'4'}.items():common+=['-e',k+'='+v]
checks=[('metadata-gdn-copy',['-m','unittest','discover','-s','tests','-p','test_target_gdn.py','-v']),('attention-metadata',['-m','unittest','discover','-s','tests','-p','test_attention_metadata.py','-v']),('m04-copy',['-m','unittest','discover','-s','tests','-p','test_m04_output_copy.py','-v']),('inventory',['-m','unittest','discover','-s','tests','-p','test_checkpoint_inventory.py','-v']),('attention',['tests/test_guarded_attention_xpu.py','--library','/opt/exl3xpu/m04/m04.so','--out','/results/attention']),('linear',['tests/test_linear_rows_xpu.py','--model','/models/checkpoint','--out','/results/linear'])]
with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()==['b70-chatui'], 'GPU worker still active'
 try:
  for name,args in checks:
   command=common+['--entrypoint','python',image,'-u',*args]
   state['checks'][name]=dict(status='RUNNING',command=command,started_unix=time.time());save();print('START',name,flush=True)
   with (root/(name+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
   state['checks'][name].update(status='PASS',finished_unix=time.time());save();print('PASS',name,flush=True)
  state['status']='PASS_NEW_ABI_KERNEL_GATES'
 except BaseException as e:state.update(status='FAILED',error=repr(e));raise
 finally:state['finished_unix']=time.time();save()

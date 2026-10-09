#!/usr/bin/env python3
"""Serialize GPU experiment and restore the qualified MTP3 service on every exit."""
import argparse
import fcntl
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'scripts'))
from exl3_candidate_worker import profile
spec=importlib.util.spec_from_file_location('qualified_restore',REPO/'scripts/run-exl3-review-micro.py')
restore=importlib.util.module_from_spec(spec);spec.loader.exec_module(restore)
p=argparse.ArgumentParser();p.add_argument('--script',nargs='+',default=['test_numerics.py']);p.add_argument('--name',default='xpu-numerics');a=p.parse_args()
root=REPO/'benchmark-results/dflash2-fp16-v031-20261009'
production=json.loads((REPO/'config/experiments/vllm-031-upgrade/production_image.json').read_text())['image_id']
candidate=subprocess.check_output(['docker','image','inspect','local/b70-dflash2-fp16:v031-test','--format','{{.Id}}'],text=True).strip()
cfg=profile(production)
state=dict(status='RUNNING',production_image=production,candidate_image=candidate,started_unix=time.time(),commands=[],source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
def save():(root/(a.name+'-campaign.json')).write_text(json.dumps(state,indent=2)+'\n')
with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True)
  assert subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()==['b70-chatui']
  cmd=['docker','run','--rm','--name','b70-dflash2-numerics','--network','none','--device','/dev/dri','--shm-size','4g',
       '-v',str(REPO/'benchmarks/experiments/dflash2-fp16-v031')+':/experiment:ro','-v',str(root)+':/results',
       '-v',str(Path.home()/'.cache/exl3xpu/incoai-Qwen3.8-27B-DFlash2-dedf8df6')+':/draft:ro',
       '-v',str(Path.home()/'.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw')+':/models/checkpoint:ro',
       '-v',str(REPO/'benchmark-results/exl3-optimized-quality-v2/engine-config.json')+':/engine-config.json:ro']
  for k,v in {**cfg['env'],'ZE_AFFINITY_MASK':'0','OMP_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','PYTHONPATH':'/experiment:/opt/b70-runtime'}.items():cmd+=['-e',k+'='+v]
  for script in a.script:
   name=Path(script).stem
   command=cmd+['--entrypoint','python',candidate,'-u','/experiment/'+script,'--device','xpu','--out','/results/'+a.name+'-'+name+'.json']
   entry=dict(script=script,command=command,status='RUNNING');state['commands'].append(entry);save()
   with (root/(a.name+'-'+name+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
   entry['status']='PASS';save()
  assert all(hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==digest for name,digest in state['source_sha256'].items()), 'diagnostic source changed during campaign'
  state['status']='PASS'
 except BaseException as e:state.update(status='FAILED',error=repr(e));raise
 finally:
  subprocess.run(['docker','rm','-f','b70-dflash2-numerics'],capture_output=True)
  subprocess.run(['systemctl','--user','start','b70-qwen38-vllm.service'],check=True)
  restore.wait_for_qualified_service(production,timeout=900)
  state.update(service_restored=True,finished_unix=time.time());save()

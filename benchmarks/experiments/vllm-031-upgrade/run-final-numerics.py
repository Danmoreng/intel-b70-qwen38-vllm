#!/usr/bin/env python3
"""Sequentially requalify the frozen final bytes, keeping all failed attempts."""
import json,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];EXPERIMENT=Path(__file__).parent
root=REPO/'benchmark-results/vllm-031-upgrade-20261009';release=json.loads((REPO/'config/experiments/vllm-031-upgrade/production_image.json').read_text());image=release['image_id'];q=root/'quality-final'
state=dict(status='RUNNING',image_id=image,commands=[],started_unix=time.time())
def save():(root/'final-numerics-controller.json').write_text(json.dumps(state,indent=2)+'\n')
def command(name,cmd):
 d=dict(name=name,status='RUNNING',argv=cmd,started_unix=time.time());state['commands'].append(d);save();print('START',name,flush=True)
 with (root/(name+'-controller.log')).open('w') as log:subprocess.run(cmd,check=True,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
 d.update(status='PASS',finished_unix=time.time());save();print('PASS',name,flush=True)
try:
 command('kernels-final',[sys.executable,str(EXPERIMENT/'run-kernel-gates.py'),'--image',image,'--out',str(root/'kernels-final')])
 command('quality-final',[sys.executable,str(EXPERIMENT/'run-quality.py'),'--out',str(q),'--capture-only'])
 cmd=['docker','run','--rm','--network','none','-v',str(q)+':/results','-v',str(EXPERIMENT)+':/upgrade:ro','-v',str(REPO/'benchmarks/experiments/quantization-reference')+':/scripts:ro','-v',str(REPO/'benchmark-results/exl3-optimized-quality-v2/short/target-optimized')+':/old-short:ro','-v',str(REPO/'benchmark-results/exl3-long-quality-v3/bf16')+':/long-reference:ro','-v',str(REPO/'benchmarks/results/exl3-review-release-v2/assessment.json')+':/old-assessment.json:ro','-e','PYTHONPATH=/scripts','--entrypoint','python',image,'/upgrade/review-quality.py','--root','/results']
 command('cross-version-final',cmd)
 command('replay-final',[sys.executable,str(EXPERIMENT/'run-replay.py'),'--out',str(q)])
 state['status']='COMPLETE_FINAL_NUMERICS_REQUIRES_CROSS_VERSION_REVIEW'
except BaseException as e:state.update(status='FAILED',error=repr(e));raise
finally:state['finished_unix']=time.time();save()

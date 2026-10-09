#!/usr/bin/env python3
"""After final numerical qualification: contract gates then full frozen matrix."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];EXP=Path(__file__).parent;root=REPO/'benchmark-results/vllm-031-upgrade-20261009';cfg=REPO/'config/experiments/vllm-031-upgrade'
r=json.loads((cfg/'production_image.json').read_text());n=json.loads((root/'quality-final/numerical-review.json').read_text());assert n['status']=='PASS_REVIEWED_CROSS_VERSION_NUMERICS' and n['image_id']==r['image_id']
state=dict(status='RUNNING',image_id=r['image_id'],policy_sha256=r['policy_sha256'],commands=[],started_unix=time.time(),sources_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),EXP/'run-operations.py',REPO/'scripts/run-exl3-review-serving.py',REPO/'scripts/current-profile-benchmark.py']})
def save():(root/'final-serving-controller.json').write_text(json.dumps(state,indent=2)+'\n')
def command(name,cmd):
 d=dict(name=name,status='RUNNING',argv=cmd,started_unix=time.time());state['commands'].append(d);save();print('START',name,flush=True)
 with (root/(name+'-controller.log')).open('w') as log:subprocess.run(cmd,check=True,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
 d.update(status='PASS',finished_unix=time.time());save();print('PASS',name,flush=True)
try:
 command('operations-final',[sys.executable,str(EXP/'run-operations.py'),'--out',str(root/'operations-final')])
 assert json.loads((root/'operations-final/operations.json').read_text())['status']=='PASS_NEW_IMAGE_OPERATIONAL_GATES'
 command('serving-final',[sys.executable,str(REPO/'scripts/run-exl3-review-serving.py'),'--release-dir',str(cfg),'--image-receipt',str(root/'metadata-image-r2/image.json'),'--fixture-root',str(REPO.parent/'intel-b70-qwen38-vllm/benchmark-results/meaningful-full-profile/run-20260923-201101-w0.00'),'--out',str(root/'serving-final'),'--leave-offline'])
 state['status']='COMPLETE_FINAL_CONTRACT_AND_SERVING_MATRIX'
except BaseException as e:state.update(status='FAILED',error=repr(e));raise
finally:state['finished_unix']=time.time();save()

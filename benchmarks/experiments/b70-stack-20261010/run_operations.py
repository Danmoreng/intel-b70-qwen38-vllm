#!/usr/bin/env python3
"""Exercise the unchanged serving contract exclusively on the new image."""
import argparse,fcntl,gzip,hashlib,importlib.util,json,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO/'scripts'))
from exl3_candidate_worker import Worker,BASE,sha

def module(name,file):
 s=importlib.util.spec_from_file_location(name,REPO/'scripts'/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
Mix=module('stack_mixed','run-exl3-mixed-qualification.py')
D=module('upgrade_contract','run-exl3-contract-diagnostics.py');O=module('upgrade_operations','run-exl3-final-operations.py');S=module('upgrade_serving','run-exl3-review-serving.py')
p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
r=json.loads((REPO/'config/experiments/vllm-031-upgrade/production_image.json').read_text());image=subprocess.check_output(['docker','image','inspect',a.image,'--format','{{.Id}}'],text=True).strip()
restore=module('operations_restore','run-exl3-review-micro.py');baseline=r['image_id']
state=dict(status='RUNNING',image_id=image,reference_policy_sha256=r['policy_sha256'],cases={},started_unix=time.time())
def save():(root/'operations.json').write_text(json.dumps(state,indent=2)+'\n')
with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
  assert subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()==['b70-chatui']
  w=Worker(image,root/'serving','b70-stack-operations',env={'EXL3_MIGRATION_TRACE':'/results/trace'})
  try:
   w.start();state['loader']=S.loader(w.root);save()
   cases=[('api',lambda x:D.smoke(x,262144,{'image':32,'video':4})),('media',lambda x:D.media_case(x,{'image':32,'video':4})),*[(name,lambda x,n=name:D.operational(n,x,True)) for name in ['c4-long','c16-long','image-long','extension-abort-long']]]
   for name,fn in cases:
    print('START',name,flush=True);destination=root/name;destination.mkdir();state['cases'][name]=fn(destination)
    if name=='near-limit':assert state['cases'][name]['native']['preemptions']==0
    save();print('PASS',name,flush=True)
   panel=json.loads(gzip.decompress((REPO/'benchmark-results/exl3-optimized-quality-v2/performance-panel.json.gz').read_bytes()))
   for n in [1,3]:
    print('START actual-prefill-decode-overlap',n,flush=True)
    state['cases']['mixed-'+str(n+1)]=Mix.mixed(panel,n,root/('mixed-'+str(n+1)),0);save()
   pictures=[O.unique_red(i) for i in range(32)];before=D.R.snapshot(BASE)
   response=D.R.http(BASE,'/v1/chat/completions',D.chat(pictures+[dict(type='text',text='What main color is shared by all32 pictures? Reply with one word.')]),timeout=1800)
   _,native=D.R.wait_accounted(BASE,before,response['usage'])
   assert 'red' in response['choices'][0]['message']['content'].lower() and native['preemptions']==0
   assert response['usage']['prompt_tokens']>100000
   state['cases']['independent32-max-area-images']=dict(status='PASS',usage=response['usage'],native=native,unique_images=32,pixels_each=4194304,fixture_sha256=[hashlib.sha256(x['image_url']['url'].encode()).hexdigest() for x in pictures]);save()
   subprocess.run([sys.executable,str(REPO/'scripts/collect-review-runtime.py'),'--container',w.name,'--output',str(root/'runtime-sources')],check=True)
  finally:w.stop()
  aborted=sum(p.read_text().count('FINISHED_ABORTED') for p in (w.root/'trace').glob('*') if p.is_file() and p.suffix in ('.json','.jsonl'))
  assert aborted>0;state['confirmed_abort_trace_occurrences']=aborted
  w=Worker(image,root/'restarted','b70-stack-operations')
  try:
   w.start();response=D.R.http(BASE,'/v1/chat/completions',D.chat('Reply with the capital of France, one word.'),timeout=180)
   assert 'paris' in response['choices'][0]['message']['content'].lower()
   state['cases']['worker-restart']=dict(status='PASS',loader=S.loader(w.root),reply=response['choices'][0]['message']['content'])
  finally:w.stop()
  state['status']='PASS_NEW_IMAGE_OPERATIONAL_GATES'
 except BaseException as e:state.update(status='FAILED',error=repr(e));raise
 finally:
  subprocess.run(['docker','rm','-f','b70-stack-operations'],capture_output=True,timeout=60)
  subprocess.run(['systemctl','--user','start','b70-qwen38-vllm.service'],check=True,timeout=90)
  state['restoration']=restore.wait_for_qualified_service(baseline,timeout=900)
  state['finished_unix']=time.time();save();print('RESTORED_QUALIFIED_PRODUCTION',flush=True)

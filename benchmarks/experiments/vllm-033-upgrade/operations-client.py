#!/usr/bin/env python3
"""Replay established API/media/context gates against the isolated 0.33 server."""
import argparse,hashlib,importlib.util,json,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'scripts'))
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
D=module('upgrade_api_contract',REPO/'scripts/run-exl3-contract-diagnostics.py')
O=module('upgrade_api_media',REPO/'scripts/run-exl3-final-operations.py')
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path)
p.add_argument('--tokenizer-baseline',type=Path);p.add_argument('--tokenizer-fixtures',type=Path);a=p.parse_args()
assert bool(a.tokenizer_baseline)==bool(a.tokenizer_fixtures)
root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
state={'status':'RUNNING','cases':{},'started_unix':time.time()}
def save():(root/'operations.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 if a.tokenizer_baseline:
  subprocess.run([sys.executable,str(Path(__file__).parent/'check-tokenizer.py'),'--base',D.BASE,'--fixtures',str(a.tokenizer_fixtures),'--compare',str(a.tokenizer_baseline),'--out',str(root/'tokenizer-pair.json')],check=True,timeout=600)
  state['cases']['paired-tokenizer']={'status':'PASS_ALL124_RENDERED_TOKEN_IDS_IDENTICAL'};save()
 cases=[('api',lambda x:D.smoke(x,262144,{'image':32,'video':4})),
        ('media',lambda x:D.media_case(x,{'image':32,'video':4})),
        ('near-limit',lambda x:D.long_case('near-limit',x,262144)),
        *[(name,lambda x,n=name:D.operational(n,x,True)) for name in ('c4-long','c16-long','image-long','extension-abort-long')]]
 for name,fn in cases:
  destination=root/name;destination.mkdir();print('START',name,flush=True)
  state['cases'][name]=fn(destination);save();print('PASS',name,flush=True)
 subprocess.run([sys.executable,str(Path(__file__).parent/'check-structured.py'),'--base',D.BASE,'--out',str(root/'structured.json')],check=True,timeout=420)
 state['cases']['structured-json-and-forced-tool']={'status':json.loads((root/'structured.json').read_text())['status']}
 save()
 pictures=[O.unique_red(i) for i in range(32)];before=D.R.snapshot(D.BASE)
 response=D.R.http(D.BASE,'/v1/chat/completions',D.chat(pictures+[{'type':'text','text':'What main color is shared by all 32 pictures? Reply with one word.'}]),timeout=1800)
 _,native=D.R.wait_accounted(D.BASE,before,response['usage'])
 assert 'red' in response['choices'][0]['message']['content'].lower()
 assert response['usage']['prompt_tokens']>100000 and native['preemptions']==0
 state['cases']['32-unique-max-area-images']={'status':'PASS','usage':response['usage'],'native':native,
  'pixels_each':4194304,'unique_images':32,'fixture_sha256':[hashlib.sha256(x['image_url']['url'].encode()).hexdigest() for x in pictures]}
 save()
 cmd=[sys.executable,str(REPO/'benchmarks/results/ram-kv-layout-20261108/check-production.py'),
      '--base',D.BASE,'--out',str(root/'natural-ram-restore'),'--natural-restore']
 with (root/'natural-ram-restore.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=1080)
 restore=json.loads((root/'natural-ram-restore/natural-restore.json').read_text());assert restore['passed']
 state['cases']['natural-cpu-restore']={'status':'PASS','restore_delta':restore['restore_delta'],
  'correct_markers':restore['correct_markers'],'same_answer':restore['same_answer']}
 save()
 cmd=[sys.executable,str(REPO/'scripts/test-kv-cache-candidate.py'),'--base',D.BASE,
  '--mode','admission','--tokens','200000','--secondary-tokens','131000','--count','2','--out',str(root/'admission.json')]
 with (root/'admission.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=1500)
 admission=json.loads((root/'admission.json').read_text());assert admission['validation']=='passed'
 state['cases']['completion-reserved-admission']={'status':'PASS','preemptions':admission['metricDelta'].get('vllm:num_preemptions_total'),'prompt_tokens':[r['usage']['prompt_tokens'] for r in admission['rows']]}
 state['status']='PASS_ESTABLISHED_OPERATIONAL_AND_NEW_RAM_GATES'
except BaseException as e:state.update(status='FAILED',error=repr(e));raise
finally:state['finished_unix']=time.time();save()

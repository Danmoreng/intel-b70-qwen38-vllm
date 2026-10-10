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
candidate=json.loads((REPO/'config/experiments/b70-stack-20261010/production_image.json').read_text())
assert image==candidate['image_id']
state=dict(status='RUNNING',image_id=image,policy_sha256=candidate['policy_sha256'],reference_policy_sha256=r['policy_sha256'],cases={},started_unix=time.time())
def save():(root/'operations.json').write_text(json.dumps(state,indent=2)+'\n')
snapshot=root/'diagnostic-sources';snapshot.mkdir()
for name in ['inventory.py','profile_capture.py']:(snapshot/name).write_bytes((Path(__file__).parent/name).read_bytes())
state['diagnostic_source_sha256']={p.name:sha(p) for p in snapshot.glob('*.py')}
library=json.loads((REPO/'benchmark-results/b70-stack-20261010/E01-oneDNN-build/library-receipt.json').read_text())
def worker(destination,trace=False):
 env={'VLLM_SERVER_DEV_MODE':'1','PYTHONPATH':'/experiment:/opt/b70-runtime'}
 if trace:env['EXL3_MIGRATION_TRACE']='/results/trace'
 result=Worker(image,destination,'b70-stack-operations',env=env,binds=[(snapshot,'/experiment')])
 result.profile['argv']+=['--worker-extension-cls','inventory.WorkerExtension']
 return result
def inventory(w,label):
 D.R.http(BASE,'/collective_rpc',dict(method='inventory',args=[],kwargs=dict(label=label,expected_onednn_sha256=library['library_sha256'],expected_onednn_version=library['version'])))
 result=json.loads((w.root/('inventory-'+label+'.json')).read_text())
 assert all('bfloat16' not in key for role in result['registered_tensor_dtypes'].values() for key in role)
 return {key:result[key] for key in ['num_blocks','kv_pool_bytes','torch_allocated_bytes','torch_reserved_bytes','registered_tensor_dtypes']}
with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
  assert subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()==['b70-chatui']
  # A metadata-only child has a new compiler-cache namespace. Record its
  # first-touch allocator profile, then compare operational admission with
  # the qualified workload's normal warm-compiler 200-block pool. Prefix-cache
  # cold/warm definitions in the actual cases remain unchanged.
  w=worker(root/'compiler-prime')
  try:
   w.start();state['compiler_first_start_inventory']=inventory(w,'compiler-first-start')
   state['compiler_first_start_loader']=S.loader(w.root)
   state['compiler_priming_scope']='Startup and read-only inventory only; no serving performance measurement. First-start pool is retained and accounted separately, never forced with a KV-memory override.'
   save()
  finally:w.stop()
  w=worker(root/'serving',trace=True)
  try:
   w.start();state['loader']=S.loader(w.root);state['initial_inventory']=inventory(w,'initial');save()
   assert state['initial_inventory']['num_blocks']==200 and state['initial_inventory']['kv_pool_bytes']==11141120000
   cases=[('api',lambda x:D.smoke(x,262144,{'image':32,'video':4})),('media',lambda x:D.media_case(x,{'image':32,'video':4})),*[(name,lambda x,n=name:D.operational(n,x,True)) for name in ['c4-long','c16-long','image-long','extension-abort-long']]]
   for name,fn in cases:
    print('START',name,flush=True);destination=root/name;destination.mkdir();state['cases'][name]=fn(destination)
    if name=='near-limit':assert state['cases'][name]['native']['preemptions']==0
    save();print('PASS',name,flush=True)
   old=json.loads((REPO/'benchmark-results/vllm-031-upgrade-20261009/operations-final/operations.json').read_text())['cases']['c16-long']
   new=state['cases']['c16-long']
   assert new['request_sha256']==old['request_sha256']
   excess=lambda x:max(0,x['native']['prefill_tokens']-x['native']['prompt_tokens']+x['native']['cached_tokens'])
   state['c16_pressure_comparison']=dict(baseline_preemptions=old['native']['preemptions'],candidate_preemptions=new['native']['preemptions'],baseline_prefill_recompute_excess=excess(old),candidate_prefill_recompute_excess=excess(new))
   save()
   assert new['native']['preemptions']<=old['native']['preemptions'] and excess(new)<=excess(old)
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
  w=worker(root/'restarted')
  try:
   w.start();state['restarted_inventory']=inventory(w,'restarted');save()
   assert state['restarted_inventory']['num_blocks']==200 and state['restarted_inventory']['kv_pool_bytes']==11141120000
   response=D.R.http(BASE,'/v1/chat/completions',D.chat('Reply with the capital of France, one word.'),timeout=180)
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

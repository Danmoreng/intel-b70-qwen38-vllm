#!/usr/bin/env python3
"""Exclusive v0.33 qualification; restore the running production service in finally."""
import argparse,datetime as dt,fcntl,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'scripts'))
from exl3_candidate_worker import MODEL,profile
from ram_kv_release import load_overlay,serving_arguments
SERVICE='b70-qwen38-vllm.service'
NAME='b70-v033-candidate'
KERNEL='b70-v033-kernels'
BASE='http://127.0.0.1:8082'
CAP=next(Path('/sys/bus/pci/devices/0000:03:00.0/hwmon').glob('*/power1_cap'))

def run(cmd,**kw):return subprocess.run(cmd,check=True,**kw)
def ready(base,timeout=900):
 end=time.monotonic()+timeout
 while time.monotonic()<end:
  try:
   with urllib.request.urlopen(base+'/v1/models',timeout=3) as r:return json.load(r)
  except OSError:time.sleep(2)
 raise TimeoutError('Readiness deadline: '+base)
def idle(base):
 raw=urllib.request.urlopen(base+'/metrics',timeout=10).read().decode()
 for line in raw.splitlines():
  if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{')):
   if float(line.split()[-1]):raise RuntimeError('Model is busy: '+line)
def time_guard():
 now=dt.datetime.now(ZoneInfo('Europe/Berlin'))
 if (now.hour,now.minute)>=(22,0):raise RuntimeError('Refusing another major stage after22:00')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--image',required=True);p.add_argument('--out',required=True,type=Path)
 p.add_argument('--production-repo',required=True,type=Path)
 p.add_argument('--fixture-root',required=True,type=Path)
 p.add_argument('--stages',default='kernels,serving')
 p.add_argument('--only',action='append',default=[],help='Frozen serving scenario/group filter for matched follow-up measurements')
 p.add_argument('--tokenizer-baseline',type=Path);p.add_argument('--tokenizer-fixtures',type=Path)
 a=p.parse_args();assert bool(a.tokenizer_baseline)==bool(a.tokenizer_fixtures)
 time_guard();root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
 image=json.loads(subprocess.check_output(['docker','image','inspect',a.image],text=True))[0]['Id']
 baseline=json.loads(subprocess.check_output(['docker','inspect','b70-qwen38-vllm'],text=True))[0]
 assert image!=baseline['Image']
 assert int(CAP.read_text())==180000000
 cfg=profile(image);assert cfg['env']['EXL3_TARGET_RUNTIME']=='vllm033'
 release=json.loads((a.production_repo/'config/production_image.json').read_text())
 overlay=load_overlay(a.production_repo,release,{})
 assert overlay['environment']=={'VLLM_KV_CACHE_LAYOUT':'LBNHC'}
 runtime=Path(__file__).parent/'runtime'
 env={**cfg['env'],**overlay['environment'],'EXL3_SDPA_CACHE_CAPACITY':'64','HF_HUB_OFFLINE':'1',
      'PYTHONPATH':'/opt/b70-runtime','ZE_AFFINITY_MASK':'0','EXL3_LOADER_REPORT_DIR':'/results/loader',
      'EXL3_LIB':'/opt/exl3xpu/exl3xpu/_C.so'}
 state={'status':'RUNNING','image_id':image,'baseline_image_id':baseline['Image'],
        'started_utc':dt.datetime.now(dt.UTC).isoformat(),'stages':{},'power_cap_w':180,'serving_scenario_filters':a.only}
 def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
 def stop(name,folder):
  subprocess.run(['docker','stop','--time','90',name],capture_output=True,timeout=105)
  logs=subprocess.run(['docker','logs',name],capture_output=True,text=True)
  (root/folder).write_text(logs.stdout+logs.stderr)
  subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=60)
 def common(name,folder):
  folder.mkdir(parents=True,exist_ok=True)
  cmd=['docker','run','--rm','--name',name,'--network','none','--device','/dev/dri','--shm-size','14g',
       '--memory','23g','--memory-swap','23g','--label','org.local.b70.release-status=EXPERIMENT',
       '-v',str(MODEL)+':/models/checkpoint:ro','-v',str(runtime.resolve())+':/opt/b70-runtime:ro',
       '-v',str(folder)+':/results']
  for k,v in env.items():cmd+=['-e',k+'='+v]
  return cmd
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
  signal.signal(sig,lambda n,f:(_ for _ in ()).throw(KeyboardInterrupt('signal '+str(n))))
 lock_path=Path('/tmp/b70-v033-qualification.lock')
 with lock_path.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle('http://127.0.0.1:8081');save()
  run(['systemctl','--user','stop',SERVICE],timeout=120)
  try:
   for name in (NAME,KERNEL):
    assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0,'Existing worker '+name
   for stage in a.stages.split(','):
    time_guard();state['stages'][stage]={'status':'RUNNING'};save();print('START',stage,flush=True)
    if stage in ('kernels','linear'):
     checks=[('native-artifact',['scripts/verify_native_artifact.py']),
             ('gdn-semantics-and-graphs',['-m','unittest','discover','-s','tests','-p','test_target_gdn.py','-v']),
             ('attention-metadata',['-m','unittest','discover','-s','tests','-p','test_attention_metadata.py','-v']),
             ('m04-copy',['-m','unittest','discover','-s','tests','-p','test_m04_output_copy.py','-v']),
             ('checkpoint-inventory',['-m','unittest','discover','-s','tests','-p','test_checkpoint_inventory.py','-v']),
             ('attention-original-tolerances',['tests/test_guarded_attention_xpu.py','--library','/opt/exl3xpu/m04/m04.so','--out','/results/attention']),
             ('linear-original-tolerances',['tests/test_linear_rows_xpu.py','--model','/models/checkpoint','--out','/results/linear'])]
     if stage=='linear':checks=[x for x in checks if x[0]=='linear-original-tolerances']
     for label,args in checks:
      print('CHECK',label,flush=True)
      command=common(KERNEL,root/'kernels')+['--entrypoint','python',image,'-u',*args]
      with (root/(label+'.log')).open('w') as log:run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
    elif stage=='replay':
     folder=root/'replay';folder.mkdir()
     (folder/'panel.json.gz').write_bytes((a.production_repo/'benchmark-results/exl3-optimized-quality-v2/performance-panel.json.gz').read_bytes())
     (folder/'config.json').write_bytes((a.production_repo/'benchmark-results/exl3-optimized-quality-v2/engine-config.json').read_bytes())
     command=common(KERNEL,root)+['-v',str(Path(__file__).parent.resolve())+':/measure:ro',
      '-e','PYTHONPATH=/measure:/opt/b70-runtime',
      '-e','B70_REPLAY_IMAGE_SCOPE=vLLM0.33 candidate, original numerical tolerances',
      '--entrypoint','python',image,'-u','/measure/run_state_replay.py',
      '--panel','/results/replay/panel.json.gz','--config','/results/replay/config.json',
      '--out','/results/replay/observations']
     with (root/'replay.log').open('w') as log:run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
     results=[json.loads(f.read_text()) for f in (folder/'observations/steps').glob('*-result.json')]
     assert len(results)==4 and all(r['status']=='PASS_LOCAL_ATTENTION_ORIGINAL_TOLERANCE' for r in results)
     assert any(r['graph_mode']=='FULL' for r in results) and any(r['label']=='long-prefill' for r in results)
    elif stage in ('startup','serving','operations'):
     folder=root/'worker';cmd=common(NAME,folder)
     # The API stays on loopback and needs no external network.
     cmd[cmd.index('--network')+1]='bridge'
     cmd[cmd.index('--rm'):cmd.index('--rm')+1]=['-d']
     cmd+=['-p','127.0.0.1:8082:8000']
     cache=root/'cache'
     for key,target in [('vllm','/root/.cache/vllm'),('triton','/root/.triton/cache'),('neo_compiler_cache','/root/.cache/neo_compiler_cache')]:
      path=cache/key;path.mkdir(parents=True,exist_ok=True);cmd+=['-v',str(path)+':'+target]
     argv=serving_arguments(cfg['argv'][1:],overlay)+['--enable-per-request-metrics','--enable-prompt-tokens-details']
     cmd+=['--entrypoint','/opt/venv/bin/vllm',image,*argv]
     (root/'launch.json').write_text(json.dumps({'command':cmd,'profile':cfg,'overlay':overlay},indent=2)+'\n')
     run(cmd,stdout=subprocess.DEVNULL,timeout=60)
     end=time.monotonic()+900
     while time.monotonic()<end:
      current=json.loads(subprocess.check_output(['docker','inspect',NAME],text=True))[0]
      if not current['State']['Running']:raise RuntimeError('Candidate exited during startup; see worker.log')
      try:
       with urllib.request.urlopen(BASE+'/v1/models',timeout=3) as response:models=json.load(response)
       break
      except OSError:time.sleep(2)
     else:raise TimeoutError('Candidate startup')
     (root/'models.json').write_text(json.dumps(models,indent=2)+'\n');print('CANDIDATE_READY',flush=True)
     body={'model':'Qwen3.8-27B','messages':[{'role':'user','content':'Count from 1 to 100, separated by commas.'}],
           'temperature':0,'max_tokens':64,'chat_template_kwargs':{'enable_thinking':False}}
     request=urllib.request.Request(BASE+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
     with urllib.request.urlopen(request,timeout=180) as response:smoke=json.load(response)
     (root/'decode-smoke.json').write_text(json.dumps(smoke,indent=2)+'\n')
     assert smoke['usage']['completion_tokens']==64 and smoke['choices'][0]['message']['content']
     if stage=='serving':
      command=[sys.executable,str(REPO/'scripts/current-profile-benchmark.py'),'--base',BASE,'--container',NAME,
       '--expected-max-num-seqs','16','--fixture-root',str(a.fixture_root.resolve()),'--legacy-prefix-namespace',
       '--output-root',str(root/'serving'),'--execute']
      for selected in a.only:command+=['--only',selected]
      with (root/'serving.log').open('w') as log:run(command,stdout=log,stderr=subprocess.STDOUT,timeout=5700)
     elif stage=='operations':
      command=[sys.executable,str(Path(__file__).parent/'operations-client.py'),'--out',str(root/'operations')]
      if a.tokenizer_baseline:command+=['--tokenizer-baseline',str(a.tokenizer_baseline.resolve()),'--tokenizer-fixtures',str(a.tokenizer_fixtures.resolve())]
      with (root/'operations.log').open('w') as log:run(command,stdout=log,stderr=subprocess.STDOUT,timeout=5700)
     stop(NAME,'worker.log')
     if stage=='operations':
      run(cmd,stdout=subprocess.DEVNULL,timeout=60)
      ready(BASE)
      body['messages']=[{'role':'user','content':'Name the capital of France. Reply with one word.'}]
      request=urllib.request.Request(BASE+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
      with urllib.request.urlopen(request,timeout=180) as response:restart=json.load(response)
      assert 'paris' in restart['choices'][0]['message']['content'].lower()
      (root/'restart.json').write_text(json.dumps(restart,indent=2)+'\n')
      stop(NAME,'worker-restarted.log')
    else:raise ValueError(stage)
    state['stages'][stage]['status']='PASS';save();print('PASS',stage,flush=True)
   state['status']='PASS_REQUESTED_STAGES'
  except BaseException as e:state.update(status='FAILED',error=repr(e));raise
  finally:
   stop(NAME,'worker-final.log');stop(KERNEL,'kernel-final.log')
   CAP.write_text('180000000\n')
   run(['systemctl','--user','start',SERVICE],timeout=120)
   state['restored_models']=ready('http://127.0.0.1:8081')
   actual=json.loads(subprocess.check_output(['docker','inspect','b70-qwen38-vllm'],text=True))[0]
   assert actual['Image']==baseline['Image']
   state['production_restored']=True;state['finished_utc']=dt.datetime.now(dt.UTC).isoformat();save()
   print('RESTORED_PRODUCTION',flush=True)
if __name__=='__main__':main()

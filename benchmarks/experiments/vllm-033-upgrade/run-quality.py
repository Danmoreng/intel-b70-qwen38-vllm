#!/usr/bin/env python3
"""Fresh, exclusive teacher-forced reference/candidate capture; always restore service."""
import argparse,datetime as dt,json,signal,subprocess,sys,time,urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo
REPO=Path(__file__).resolve().parents[3];EXP=Path(__file__).parent
sys.path.insert(0,str(REPO/'scripts'))
from exl3_candidate_worker import MODEL,profile
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--image',required=True);p.add_argument('--out',required=True,type=Path)
p.add_argument('--production-repo',required=True,type=Path)
p.add_argument('--kinds',default='short,long');p.add_argument('--long-windows',type=int,default=4,choices=(1,4))
a=p.parse_args()
assert dt.datetime.now(ZoneInfo('Europe/Berlin')).hour<21
root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
image=json.loads(subprocess.check_output(['docker','image','inspect',a.image],text=True))[0]['Id']
baseline=json.loads(subprocess.check_output(['docker','inspect','b70-qwen38-vllm'],text=True))[0]['Image']
raw=urllib.request.urlopen('http://127.0.0.1:8081/metrics',timeout=10).read().decode()
for line in raw.splitlines():
 if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{')):assert float(line.split()[-1])==0,line
config=json.loads((a.production_repo/'benchmark-results/exl3-optimized-quality-v2/engine-config.json').read_text())
(root/'engine-config.json').write_text(json.dumps(config,indent=2)+'\n')
for kind,source in [('short','quantization-reference-20261001'),('long','exl3-long-quality-v3')]:
 path=root/kind;path.mkdir();(path/'panel.json').write_bytes((a.production_repo/'benchmark-results'/source/'panel.json').read_bytes())
state={'status':'RUNNING','image_id':image,'production_image_id':baseline,'checks':{},'started_utc':dt.datetime.now(dt.UTC).isoformat()}
def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,lambda n,f:(_ for _ in ()).throw(KeyboardInterrupt(str(n))))
name='b70-v033-quality';assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0
cfg=profile(image)
common=['docker','run','--rm','--name',name,'--network','none','--device','/dev/dri','--shm-size','8g',
 '-v',str(MODEL)+':/models/checkpoint:ro','-v',str(root)+':/results',
 '-v',str((REPO/'benchmarks/experiments/quantization-reference').resolve())+':/scripts:ro',
 '-v',str(EXP.resolve())+':/measure:ro','-v',str((REPO/'runtime').resolve())+':/opt/b70-runtime:ro']
for k,v in {**cfg['env'],'VLLM_KV_CACHE_LAYOUT':'LBNHC','HF_HUB_OFFLINE':'1','PYTHONPATH':'/measure:/scripts:/opt/b70-runtime',
 'ZE_AFFINITY_MASK':'0','EXL3_SDPA_CACHE_CAPACITY':'64','EXL3_LOADER_REPORT_DIR':'/results/loader'}.items():common+=['-e',k+'='+v]
for key,target in [('vllm','/root/.cache/vllm'),('triton','/root/.triton/cache'),('neo_compiler_cache','/root/.cache/neo_compiler_cache')]:
 path=root/'cache'/key;path.mkdir(parents=True);common+=['-v',str(path)+':'+target]
try:
 save();subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=120)
 for kind in a.kinds.split(','):
  assert kind in ('short','long')
  cmd=common+['--entrypoint','python',image,'-u','-c',
   'import native_capture; assert native_capture.__file__=="/measure/native_capture.py"; import runpy; runpy.run_path("/scripts/run_native.py",run_name="__main__")','--model','/models/checkpoint',
   '--quantization','exl3','--panel','/results/'+kind+'/panel.json','--out','/results/'+kind+'/capture',
   '--engine-config','/results/engine-config.json']
  if kind=='long':cmd+=['--windows',str(a.long_windows)]
  state['checks'][kind]={'status':'RUNNING','command':cmd};save();print('START',kind,flush=True)
  with (root/(kind+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
  state['checks'][kind]['status']='PASS';save();print('PASS',kind,flush=True)
 state['status']='PASS_ALIGNED_TEACHER_FORCED_CAPTURE'
except BaseException as e:state.update(status='FAILED',error=repr(e));raise
finally:
 subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=60)
 subprocess.run(['systemctl','--user','start','b70-qwen38-vllm.service'],check=True,timeout=120)
 end=time.monotonic()+900
 while time.monotonic()<end:
  try:
   urllib.request.urlopen('http://127.0.0.1:8081/health',timeout=3).close();break
  except OSError:time.sleep(2)
 else:raise TimeoutError('Production did not recover')
 actual=json.loads(subprocess.check_output(['docker','inspect','b70-qwen38-vllm'],text=True))[0]['Image']
 assert actual==baseline
 state['production_restored']=True;state['finished_utc']=dt.datetime.now(dt.UTC).isoformat();save()
 print('RESTORED_PRODUCTION',flush=True)

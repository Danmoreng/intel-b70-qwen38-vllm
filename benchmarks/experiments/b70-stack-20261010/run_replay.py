#!/usr/bin/env python3
"""New-image quality capture and replay against stored references only."""
import argparse,fcntl,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO/'scripts'))
from exl3_candidate_worker import MODEL,profile,sha
p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.out.resolve();assert (root/'short/candidate/summary.json').exists() and (root/'long/candidate/summary.json').exists()
r=json.loads((REPO/'config/experiments/vllm-031-upgrade/production_image.json').read_text());image=subprocess.check_output(['docker','image','inspect',a.image,'--format','{{.Id}}'],text=True).strip()
assert json.loads((root/'campaign.json').read_text())['image_id']==image
import importlib.util
spec=importlib.util.spec_from_file_location('state_restore',REPO/'scripts/run-exl3-review-micro.py');restore=importlib.util.module_from_spec(spec);spec.loader.exec_module(restore);baseline=r['image_id']
state=dict(status='RUNNING',image_id=image,reference_policy_sha256=r['policy_sha256'],reference_reused=True,commands=[],started_unix=time.time(),quality_capture_campaign_sha256=sha(root/'campaign.json'))
def save():(root/'replay-campaign.json').write_text(json.dumps(state,indent=2)+'\n')
cfg=profile(image)
cache=Path.home()/'.cache/exl3xpu/optimized-qualification'/image.removeprefix('sha256:')
common=['docker','run','--rm','--name','b70-stack-replay','--network','none','--device','/dev/dri','--memory','14g','--shm-size','4g','-v',str(MODEL)+':/models/checkpoint:ro','-v',str(MODEL)+':/exl3:ro','-v',str(REPO/'benchmarks/experiments/quantization-reference')+':/scripts:ro','-v',str(REPO/'benchmarks/experiments/vllm-031-upgrade')+':/upgrade-scripts:ro','-v',str(REPO/'benchmarks/experiments/vllm-031-upgrade/native_capture.py')+':/scripts/native_capture.py:ro','-v',str(REPO/'benchmarks/experiments/exl3-review-release-v2')+':/review:ro','-v',str(root)+':/results','-v',str(REPO/'benchmark-results/quantization-reference-20261001')+':/reference:ro','-v',str(REPO/'benchmark-results/exl3-long-quality-v3/bf16')+':/long-reference:ro','-v',str(REPO/'benchmark-results/exl3-optimized-quality-v2/short/target-optimized')+':/old-short:ro']
for k,target in [('vllm','/root/.cache/vllm'),('triton','/root/.triton/cache'),('neo_compiler_cache','/root/.cache/neo_compiler_cache')]:
 (cache/k).mkdir(parents=True,exist_ok=True);common+=['-v',str(cache/k)+':'+target]
for k,v in {**cfg['env'],'EXL3_SDPA_CACHE_CAPACITY':'64','HF_HUB_OFFLINE':'1','PYTHONPATH':'/upgrade-scripts:/scripts:/review','ZE_AFFINITY_MASK':'0','OMP_NUM_THREADS':'4','EXL3_LOADER_REPORT_DIR':'/results/loader','B70_REPLAY_IMAGE_SCOPE':'New vLLM 0.31/Torch 2.14 candidate only; unchanged numerical tolerances'}.items():common+=['-e',k+'='+v]
def command(name,args,extra=None):
 cmd=common+(extra or [])+['--entrypoint','python',image,'-u',*args]
 item=dict(name=name,command=cmd,status='RUNNING',started_unix=time.time());state['commands'].append(item);save();print('START',name,flush=True)
 with (root/(name+'.log')).open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
 item.update(status='PASS',finished_unix=time.time());save();print('PASS',name,flush=True)
with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
  assert subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()==['b70-chatui']
  command('linear-reference',['/opt/exl3xpu/tests/test_linear_rows_xpu.py','--model','/models/checkpoint','--out','/results/linear'],['-e','EXL3_LIB=/opt/exl3xpu/exl3xpu/_C.so'])
  replay=root/'replay';replay.mkdir();snapshots=replay/'diagnostic-sources';snapshots.mkdir()
  original=REPO/'benchmarks/experiments/exl3-review-20261002'
  for f in ['state_replay.py','run_state_replay.py']:shutil.copyfile(original/f,snapshots/f)
  f=snapshots/'state_replay.py';s=f.read_text();old="RUNNER_SHA256 = '174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce'";assert s.count(old)==1
  # Reviewed execute_model keeps all three guarded call sites and input variables;
  # the unchanged diagnostic already passes is_padding through forward_context.
  h=sha(REPO/'benchmark-results/vllm-031-upgrade-20261009/installed-vllm/v1/worker/gpu/model_runner.py')
  assert h=='725c47114e148ade3a26b4cc9c5f7295588fdd387afc3cb5fdb2d16513a1bb5c'
  f.write_text(s.replace(old,"RUNNER_SHA256 = '"+h+"'"))
  (replay/'panel.json.gz').write_bytes((REPO/'benchmark-results/exl3-optimized-quality-v2/performance-panel.json.gz').read_bytes());(replay/'config.json').write_bytes((root/'config.json').read_bytes())
  command('candidate-matched-state',['/replay-scripts/run_state_replay.py','--panel','/results/replay/panel.json.gz','--config','/results/replay/config.json','--out','/results/replay/observations'],['-v',str(snapshots)+':/replay-scripts:ro','-e','PYTHONPATH=/replay-scripts'])
  assert json.loads((replay/'observations/summary.json').read_text())['status']=='COMPLETE_MATCHED_STATE_DIAGNOSTIC'
  results=[json.loads(x.read_text()) for x in (replay/'observations/steps').glob('*-result.json')]
  assert len(results)==4 and all(d['status']=='PASS_LOCAL_ATTENTION_ORIGINAL_TOLERANCE' for d in results)
  assert any(d['graph_mode']=='FULL' for d in results) and any(d['label']=='long-prefill' for d in results)
  state['status']='PASS_NEW_IMAGE_MATCHED_STATE_ORIGINAL_TOLERANCES'
 except BaseException as e:state.update(status='FAILED',error=repr(e));raise
 finally:
  subprocess.run(['docker','rm','-f','b70-stack-replay'],capture_output=True,timeout=60)
  subprocess.run(['systemctl','--user','start','b70-qwen38-vllm.service'],check=True,timeout=90)
  state['restoration']=restore.wait_for_qualified_service(baseline,timeout=900)
  state['finished_unix']=time.time();save();print('RESTORED_QUALIFIED_PRODUCTION',flush=True)

"""Actual eager branch capture, independent CPU replay, guaranteed rollback."""
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO/'scripts'))
from exl3_candidate_worker import Worker,BASE,sha
IMAGE='sha256:b0b9244420a2586f8328b14aeff21d94673632ad7342032a285f3b877f2de49b'
production=json.loads((REPO/'config/experiments/vllm-031-upgrade/production_image.json').read_text())['image_id']
spec=importlib.util.spec_from_file_location('restore',REPO/'scripts/run-exl3-review-micro.py')
restore=importlib.util.module_from_spec(spec);spec.loader.exec_module(restore)
name=sys.argv[1];root=REPO/'benchmark-results/dflash2-pro-diagnostics-v031'/name;root.mkdir(exist_ok=False)
snapshot=root/'source-snapshot';snapshot.mkdir()
for directory in [Path(__file__).parent,REPO/'benchmarks/experiments/dflash2-serving-v031']:
    for p in directory.glob('*.py'):
        target=snapshot/p.name;assert not target.exists();target.write_bytes(p.read_bytes())
draft=Path.home()/'.cache/exl3xpu/incoai-Qwen3.8-27B-DFlash2-dedf8df6-fp16'
conversion=json.loads((draft/'conversion.json').read_text());assert sha(draft/'model.safetensors')==conversion['fp16_sha256']
fixture=REPO.parent/'intel-b70-qwen38-vllm/benchmark-results/meaningful-full-profile/run-20260923-201101-w0.00'
manifest=json.loads((REPO/'config/experiments/exl3-migration/full-serving-fixture-manifest.json').read_text())
assert sha(fixture/'phase-32k-c1-r1/prompt-1.txt')==manifest['files']['phase-32k-c1-r1/prompt-1.txt']
state=dict(status='RUNNING',image=IMAGE,production_image=production,started_unix=time.time(),
    sources={p.name:sha(p) for p in snapshot.glob('*.py')},diagnostic_only=True,enforce_eager=True)
def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
def api(path,data,timeout=900):
    req=urllib.request.Request(BASE+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)
def rpc(method):
    result=api('/collective_rpc',dict(method=method,args=[],kwargs={}))
    (root/(method+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(method,flush=True);return result
with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);w=None;save()
    try:
        subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
        assert subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()==['b70-chatui']
        w=Worker(IMAGE,root/'worker','b70-dflash-branches',
            env={'VLLM_SERVER_DEV_MODE':'1','PYTHONPATH':'/experiment:/opt/b70-runtime'},
            binds=[(snapshot,'/experiment'),(draft,'/draft')])
        argv=w.profile['argv'];argv[argv.index('--max-model-len')+1]='96000'
        argv[argv.index('--speculative-config')+1]=json.dumps(dict(method='dflash',model='/draft',num_speculative_tokens=7,draft_sample_method='greedy'))
        argv+=['--enforce-eager','--worker-extension-cls','branch_capture.WorkerExtension']
        w.start();rpc('inspect_dflash_setup');rpc('cache_ledger');rpc('epsilon_probe');rpc('rejection_greedy_test');rpc('install_branch_capture')
        rpc('install_round_capture');rpc('begin_round_capture');api('/reset_prefix_cache',{})
        body=dict(model='Qwen3.8-27B',messages=[dict(role='user',content=(fixture/'phase-32k-c1-r1/prompt-1.txt').read_text())],
            temperature=1.,top_p=.95,top_k=20,seed=770000,max_tokens=384,ignore_eos=True,
            chat_template_kwargs={'enable_thinking':False})
        (root/'request.json').write_text(json.dumps(body)+'\n')
        response=api('/v1/chat/completions',body);assert response['usage']['completion_tokens']==384
        (root/'response.json').write_text(json.dumps(response,indent=2)+'\n')
        result=rpc('finish_branch_capture');assert len(result['results'])==1,result
        assert len(result['results'][0]['files'])==4,result
        rpc('finish_round_capture');rpc('uninstall_round_capture')
        command=['docker','run','--rm','--entrypoint','python','-v',str(snapshot)+':/experiment:ro',
            '-v',str(root/'worker/branch-capture')+':/capture',IMAGE,'/experiment/replay_branches.py','/capture']
        with (root/'replay.log').open('w') as out:subprocess.run(command,check=True,stdout=out,stderr=subprocess.STDOUT,timeout=900)
        command=['docker','run','--rm','--entrypoint','python','-v',str(snapshot)+':/experiment:ro',
            '-v',str(root/'worker/round-capture')+':/capture:ro','-v',str(root)+':/out',
            IMAGE,'/experiment/replay_rounds.py','/capture','--output','/out/proposal-audit.json']
        with (root/'proposal-replay.log').open('w') as out:subprocess.run(command,check=True,stdout=out,stderr=subprocess.STDOUT,timeout=900)
        state['status']='PASS_ACTUAL_EAGER_BRANCH_REPLAY';print(state['status'],flush=True)
    except BaseException as error:state.update(status='FAILED',error=repr(error));raise
    finally:
        if w is not None:w.stop()
        subprocess.run(['systemctl','--user','start','b70-qwen38-vllm.service'],check=True,timeout=90)
        state['restoration']=restore.wait_for_qualified_service(production,timeout=900)
        state['finished_unix']=time.time();save()

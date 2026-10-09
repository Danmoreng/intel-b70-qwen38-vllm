"""Bounded, paired ABBA MTP3/4; no new power measurements or depth matrix."""
import fcntl
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import wave as W

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'scripts'))
from exl3_candidate_worker import Worker,BASE,sha
spec=importlib.util.spec_from_file_location('restore',REPO/'scripts/run-exl3-review-micro.py')
restore=importlib.util.module_from_spec(spec);spec.loader.exec_module(restore)
IMAGE='sha256:b0b9244420a2586f8328b14aeff21d94673632ad7342032a285f3b877f2de49b'
production=json.loads((REPO/'config/experiments/vllm-031-upgrade/production_image.json').read_text())['image_id']
root=REPO/'benchmark-results/mtp4-pro-qualification-v031-abba'
root.mkdir(exist_ok=False)
encoded=Path(__file__).with_name('panel.json.gz').read_bytes()
manifest=json.loads(Path(__file__).with_name('panel-manifest.json').read_text())
assert W.sha(encoded)==manifest['panel_gzip_sha256']
assert W.sha(gzip.decompress(encoded))==manifest['panel_raw_sha256']
panel=json.loads(gzip.decompress(encoded))
assert sum(w['weight'] for w in panel['windows'])==1
state=dict(status='RUNNING',image=IMAGE,production_image=production,panel=manifest,
    order=[3,4,4,3],weights={w['name']:w['weight'] for w in panel['windows']},
    minimum_weighted_gain=.05,confidence_gate='positive paired 95% CI; 2 independent AB/BA blocks are screening only',
    warm_cold_separate=True,power_w=180,power_remeasured=False,
    pairing_reason='vLLM 0.31 and wide M04 source identity differs from historical v0.30 ABBA; only three bounded cells',
    started_unix=time.time(),arms=[])
def save(): (root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
def api(path,body):
    req=urllib.request.Request(BASE+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=300) as r:return json.load(r)
snapshot=root/'source-snapshot';snapshot.mkdir()
for f in Path(__file__).parent.glob('*.py'): (snapshot/f.name).write_bytes(f.read_bytes())
identity=REPO/'benchmarks/experiments/dflash2-serving-v031/mtp_inspect.py'
(snapshot/identity.name).write_bytes(identity.read_bytes())
state['sources']={f.name:sha(f) for f in snapshot.glob('*.py')}
with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);worker=None;save()
    try:
        subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
        assert subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()==['b70-chatui']
        for index,depth in enumerate(state['order']):
            folder=root/f'arm-{index}-mtp{depth}'
            worker=Worker(IMAGE,folder,'b70-mtp4-pro',env={'VLLM_SERVER_DEV_MODE':'1','PYTHONPATH':'/experiment:/opt/b70-runtime'},binds=[(snapshot,'/experiment')])
            argv=worker.profile['argv']
            argv[argv.index('--speculative-config')+1]=json.dumps(dict(method='mtp',num_speculative_tokens=depth))
            assert argv[argv.index('--max-model-len')+1]=='262144'
            argv+=['--worker-extension-cls','mtp_inspect.WorkerExtension']
            arm=dict(index=index,depth=depth,status='STARTING',waves=[])
            state['arms'].append(arm);save();print('START_ARM',index,depth,flush=True)
            worker.start()
            attestation=api('/collective_rpc',dict(method='inspect_mtp_setup',args=[],kwargs={}))
            (folder/'attestation.json').write_text(json.dumps(attestation,indent=2)+'\n')
            audits=list((folder/'loader').glob('loader-*.json'));assert len(audits)==1
            audit=json.loads(audits[0].read_text())
            assert audit['status']=='PASS' and audit['loaded_modules']==audit['expected_modules']==409 and audit['mtp_loaded']==audit['mtp_expected']==8
            # One untimed wave per cell with distinct salts, then clear all
            # prefixes before each measured cold/warm pair.
            for window in panel['windows']:
                for mode in ['warmup','cold','warm']:
                    if mode=='cold':api('/reset_prefix_cache',{})
                    row=W.wave(window,window['concurrency'],mode,index,depth,folder,512)
                    assert row['native']['preemptions']==0
                    if mode=='warm': assert row['native']['cached_tokens']>0
                    arm['waves'].append(row);save()
                    print('WAVE',index,window['name'],mode,row['native_request_weighted_decode_tps'],row['client_aggregate_decode_tps'],flush=True)
            arm['status']='COMPLETE';worker.stop();worker=None;save()
        state['status']='COMPLETE_BOUNDED_ABBA_SCREEN'
    except BaseException as error:state.update(status='FAILED',error=repr(error));raise
    finally:
        if worker is not None:worker.stop()
        subprocess.run(['systemctl','--user','start','b70-qwen38-vllm.service'],check=True,timeout=90)
        state['restoration']=restore.wait_for_qualified_service(production,timeout=900)
        state['finished_unix']=time.time();save()

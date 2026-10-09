"""DFlash on the identical bounded MTP panel; observational screen only."""
import argparse
import fcntl
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'scripts'))
from exl3_candidate_worker import Worker,BASE,sha
spec=importlib.util.spec_from_file_location('restore',REPO/'scripts/run-exl3-review-micro.py')
restore=importlib.util.module_from_spec(spec);spec.loader.exec_module(restore)
panel_dir=REPO/'benchmarks/experiments/mtp4-pro-qualification-v031'
spec=importlib.util.spec_from_file_location('panel_wave',panel_dir/'wave.py')
W=importlib.util.module_from_spec(spec);spec.loader.exec_module(W)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--projection',choices=['full-qkv','kv-only'],default='kv-only')
args=parser.parse_args()
IMAGE={
    'full-qkv':'sha256:ebbd9575577f65f37cfe517666a5c9abe18601b94d990fd8ad8e1e1416049c05',
    'kv-only':'sha256:c6429913722c37b0091739698086fea23721de980bd549e1ab6d7d4f0bf46c74',
}[args.projection]
production=json.loads((REPO/'config/experiments/vllm-031-upgrade/production_image.json').read_text())['image_id']
root=REPO/('benchmark-results/dflash2-w4-matched-panel-v031'+
    ('-full-qkv' if args.projection=='full-qkv' else ''));root.mkdir(exist_ok=False)
manifest=json.loads((panel_dir/'panel-manifest.json').read_text())
encoded=(panel_dir/'panel.json.gz').read_bytes()
assert W.sha(encoded)==manifest['panel_gzip_sha256']
assert W.sha(gzip.decompress(encoded))==manifest['panel_raw_sha256']
panel=json.loads(gzip.decompress(encoded))
draft=Path.home()/'.cache/exl3xpu/syvai-Qwen3.8-27B-DFlash2-W4A16-4d30ec73-fp16'
conversion=json.loads((draft/'conversion.json').read_text())
assert sha(draft/'model.safetensors')==conversion['output_sha256']
assert sha(draft/'config.json')==conversion['config_sha256']
assert json.loads((draft/'config.json').read_text())['dtype']=='float16'
state=dict(status='RUNNING',image=IMAGE,production_image=production,panel=manifest,
    weights={w['name']:w['weight'] for w in panel['windows']},waves=[],started_unix=time.time(),
    depth=7,block_size=8,max_model_len=192000,draft_sample_method='greedy',
    scope='Same frozen512-token panel as current MTP ABBA; different time/order, so comparison is observational and cannot itself qualify promotion.',
    power_w=180,power_remeasured=False,context_projection=args.projection,
    draft_checkpoint_sha256=conversion['output_sha256'],draft_config_sha256=conversion['config_sha256'])
def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
def api(path,body):
    req=urllib.request.Request(BASE+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=900) as response:return json.load(response)
snapshot=root/'source-snapshot';snapshot.mkdir()
for directory in [Path(__file__).parent,REPO/'benchmarks/experiments/dflash2-serving-v031',REPO/'benchmarks/experiments/dflash2-pro-diagnostics-v031']:
    for file in directory.glob('*.py'):
        target=snapshot/file.name;assert not target.exists();target.write_bytes(file.read_bytes())
(snapshot/'panel_wave.py').write_bytes((panel_dir/'wave.py').read_bytes())
state['sources']={p.name:sha(p) for p in snapshot.glob('*.py')}
with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);w=None;save()
    try:
        subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
        assert subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()==['b70-chatui']
        w=Worker(IMAGE,root/'worker','b70-dflash-panel',env={'VLLM_SERVER_DEV_MODE':'1','PYTHONPATH':'/experiment:/opt/b70-runtime'},
            binds=[(snapshot,'/experiment'),(draft,'/draft')])
        argv=w.profile['argv'];argv[argv.index('--max-model-len')+1]='192000'
        assert argv[argv.index('--max-num-seqs')+1]=='16'
        assert argv[argv.index('--max-num-batched-tokens')+1]=='4096'
        assert argv[argv.index('--gpu-memory-utilization')+1]=='0.965'
        assert argv[argv.index('--dtype')+1]=='float16' and argv[argv.index('--kv-cache-dtype')+1]=='fp8'
        argv[argv.index('--speculative-config')+1]=json.dumps(dict(method='dflash',model='/draft',num_speculative_tokens=7,draft_sample_method='greedy'))
        argv+=['--worker-extension-cls','w4_inspect.WorkerExtension']
        w.start()
        audits=list((w.root/'loader').glob('loader-*.json'));assert len(audits)==1
        audit=json.loads(audits[0].read_text())
        assert audit['status']=='PASS' and audit['expected_modules']==audit['loaded_modules']==401
        assert audit['mtp_expected']==audit['mtp_loaded']==0
        methods=['inspect_dflash_setup','cache_ledger','w4_numerical_probe']
        if args.projection=='kv-only':methods.append('w4_prefill_kv_probe')
        for method in methods:
            value=api('/collective_rpc',dict(method=method,args=[],kwargs={}))
            (root/(method+'.json')).write_text(json.dumps(value,indent=2)+'\n')
        for window in panel['windows']:
            for mode in ['warmup','cold','warm']:
                if mode=='cold':api('/reset_prefix_cache',{})
                row=W.wave(window,window['concurrency'],mode,0,7,root/'worker',512)
                assert row['native']['preemptions']==0
                if mode=='warm':assert row['native']['cached_tokens']>0
                state['waves'].append(row);save()
                print('DF_PANEL',window['name'],mode,row['native_request_weighted_decode_tps'],row['client_aggregate_decode_tps'],flush=True)
        state['status']='COMPLETE_MATCHED_PANEL_OBSERVATIONAL_SCREEN'
    except BaseException as error:state.update(status='FAILED',error=repr(error));raise
    finally:
        if w is not None:w.stop()
        subprocess.run(['systemctl','--user','start','b70-qwen38-vllm.service'],check=True,timeout=90)
        state['restoration']=restore.wait_for_qualified_service(production,timeout=900)
        state['finished_unix']=time.time();save()

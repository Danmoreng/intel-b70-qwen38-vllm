#!/usr/bin/env python3
"""P0: exclusive pinned MTP3 worker; diagnostics and uninstrumented controls."""
import argparse,fcntl,gzip,importlib.util,json,subprocess,sys,time,urllib.request
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO/'scripts'))
from exl3_candidate_worker import Worker,BASE,sha
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
restore=module('restore',REPO/'scripts/run-exl3-review-micro.py')
W=module('wave',REPO/'benchmarks/experiments/mtp4-pro-qualification-v031/wave.py')
def api(path,body):
    req=urllib.request.Request(BASE+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=900) as r:return json.load(r)
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
    baseline=json.loads((HERE/'baseline-manifest.json').read_text());image=baseline['production']['image_id']
    panel_dir=REPO/'benchmarks/experiments/mtp4-pro-qualification-v031'
    manifest=json.loads((panel_dir/'panel-manifest.json').read_text());encoded=(panel_dir/'panel.json.gz').read_bytes()
    assert W.sha(encoded)==manifest['panel_gzip_sha256'];assert W.sha(gzip.decompress(encoded))==manifest['panel_raw_sha256']
    panel=json.loads(gzip.decompress(encoded));windows=[w for w in panel['windows'] if w['name'] in ['source-64k','prose-32768']]
    state=dict(status='RUNNING',baseline=image,gates_sha256=sha(HERE/'gates.json'),panel=manifest,waves=[],started_unix=time.time(),
        scope='P0 diagnostics, not qualification or candidate throughput; frozen MTP3 contract unchanged')
    def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
    snapshot=root/'sources';snapshot.mkdir()
    for file in HERE.glob('*.py'):(snapshot/file.name).write_bytes(file.read_bytes())
    state['sources']={p.name:sha(p) for p in snapshot.glob('*.py')}
    def rpc(method,**kwargs):
        value=api('/collective_rpc',dict(method=method,args=[],kwargs=kwargs))
        (root/(method+'-'+str(len(state.get('rpc',[])))+'.json')).write_text(json.dumps(value,indent=2)+'\n')
        state.setdefault('rpc',[]).append(dict(method=method,kwargs=kwargs));save();print('RPC',method,flush=True);return value
    with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);worker=None;save()
        try:
            subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
            names=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
            assert names==['b70-chatui'],names
            worker=Worker(image,root/'worker',name='b70-mtp3-stack-p0',env={'EXL3_SDPA_CACHE_CAPACITY':'64','VLLM_SERVER_DEV_MODE':'1','PYTHONPATH':'/experiment:/opt/b70-runtime'},binds=[(snapshot,'/experiment')])
            argv=worker.profile['argv']
            for flag,value in [('--max-model-len','262144'),('--dtype','float16'),('--kv-cache-dtype','fp8'),('--max-num-seqs','16'),('--max-num-batched-tokens','4096'),('--gpu-memory-utilization','0.965')]:assert argv[argv.index(flag)+1]==value
            assert json.loads(argv[argv.index('--speculative-config')+1])=={'method':'mtp','num_speculative_tokens':3}
            argv+=['--worker-extension-cls','inventory.WorkerExtension']
            print('START_P0_WORKER',flush=True);worker.start();rpc('inventory');rpc('install_event_profile')
            for window in windows:
                for diagnostic in ['control','events','trace']:
                    phase=worker.root/(window['name']+'-'+diagnostic);phase.mkdir()
                    api('/reset_prefix_cache',{})
                    cold=W.wave(window,window['concurrency'],'cold',0,3,phase,256)
                    if diagnostic=='events':rpc('begin_event_profile')
                    if diagnostic=='trace':rpc('begin_bounded_trace',cycles=8,min_requests=window['concurrency'])
                    warm=W.wave(window,window['concurrency'],'warm',0,3,phase,256)
                    assert warm['native']['cached_tokens']>0
                    if diagnostic=='events':rpc('finish_event_profile',path='/results/'+window['name']+'-events.json')
                    if diagnostic=='trace':rpc('finish_kernel_trace',path='/results/'+window['name']+'-trace.json')
                    state['waves'].append(dict(window=window['name'],diagnostic=diagnostic,cold=cold,warm=warm));save()
                    print('P0_WAVE',window['name'],diagnostic,'warm_native',warm['native_request_weighted_decode_tps'],'warm_client',warm['client_aggregate_decode_tps'],flush=True)
                rpc('inventory',label=window['name'])
            state['status']='COMPLETE_P0_DIAGNOSTIC_NOT_PROMOTION'
        except BaseException as e:state.update(status='FAILED',error=repr(e));raise
        finally:
            if worker is not None:worker.stop()
            subprocess.run(['systemctl','--user','start','b70-qwen38-vllm.service'],check=True,timeout=90)
            state['restoration']=restore.wait_for_qualified_service(image,timeout=900)
            state['finished_unix']=time.time();save();print('RESTORED_QUALIFIED_PRODUCTION',flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Paired fixed-MTP3 screen; preserved windows, output budget and prefix policy."""
import argparse,fcntl,gzip,importlib.util,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from exl3_candidate_worker import Worker,BASE,sha
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
restore=module('screen_restore',REPO/'scripts/run-exl3-review-micro.py')
W=module('screen_wave',REPO/'benchmarks/experiments/mtp4-pro-qualification-v031/wave.py')
D=module('screen_capacity',REPO/'scripts/run-exl3-contract-diagnostics.py')
def main():
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--m04',type=Path);p.add_argument('--blocks',type=int,default=3);p.add_argument('--capacity',action='store_true')
    p.add_argument('--protected',action='store_true')
    p.add_argument('--onednn-receipt',type=Path);a=p.parse_args()
    baseline=json.loads((HERE/'baseline-manifest.json').read_text())['production']['image_id']
    image=subprocess.check_output(['docker','image','inspect',a.image,'--format','{{.Id}}'],text=True).strip()
    root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
    folder=REPO/'benchmarks/experiments/mtp4-pro-qualification-v031';manifest=json.loads((folder/'panel-manifest.json').read_text())
    assert sha(folder/'panel.json.gz')==manifest['panel_gzip_sha256']
    panel=json.loads(gzip.decompress((folder/'panel.json.gz').read_bytes()))
    windows=[next(w for w in panel['windows'] if w['name']==n) for n in ['source-32k','source-64k','prose-32768']]
    if a.protected:
        protected_path=REPO/'benchmark-results/exl3-optimized-quality-v2/performance-panel.json.gz'
        assert sha(protected_path)=='e82311dbf9607f5b740df9e0d7305e50babd62c74d2e42f9171c8c1d59b976af'
        protected_panel=json.loads(gzip.decompress(protected_path.read_bytes()))
        for name in ['code-4096','code-131072']:
            w=next(w for w in protected_panel['windows'] if w['name']==name);windows.append({**w,'concurrency':1})
        w=next(w for w in panel['windows'] if w['name']=='source-32k');windows.append({**w,'name':'source-32k-c4-protected','concurrency':4})
    # The frozen source window has its own stable name; assert it instead of
    # substituting text or retokenizing if the panel contract differs.
    state=dict(status='RUNNING',baseline=baseline,candidate=image,blocks=[],capacity={},gates_sha256=sha(HERE/'gates.json'),
        panel=manifest,protected_panel_sha256='e82311dbf9607f5b740df9e0d7305e50babd62c74d2e42f9171c8c1d59b976af' if a.protected else None,output_budget=512,started_unix=time.time(),scope='Fixed-MTP3 paired candidate screen, not full release qualification')
    def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
    snapshot=root/'diagnostic-sources';snapshot.mkdir()
    for name in ['inventory.py','profile_capture.py']:(snapshot/name).write_bytes((HERE/name).read_bytes())
    state['diagnostic_source_sha256']={p.name:sha(p) for p in snapshot.glob('*.py')}
    with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);worker=None;save()
        try:
            subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
            assert subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()==['b70-chatui']
            for block in range(a.blocks):
                record=dict(index=block,order=['baseline','candidate'] if block%2==0 else ['candidate','baseline'],arms={});state['blocks'].append(record);save()
                for role in record['order']:
                    destination=root/f'block-{block}-{role}'
                    env={'EXL3_SDPA_CACHE_CAPACITY':'64','VLLM_SERVER_DEV_MODE':'1','PYTHONPATH':'/experiment:/opt/b70-runtime'};binds=[(snapshot,'/experiment')]
                    if role=='candidate' and a.m04:
                        env.update(EXL3_M04_LIBRARY='/candidate-m04.so',EXL3_M04_LIBRARY_SHA256=sha(a.m04))
                        binds.append((a.m04,'/candidate-m04.so'))
                    worker=Worker(baseline if role=='baseline' else image,destination,'b70-stack-screen',env=env,binds=binds)
                    worker.profile['argv']+=['--worker-extension-cls','inventory.WorkerExtension']
                    worker.start();arm=[];record['arms'][role]=arm;save()
                    kwargs=dict(label='initial')
                    if role=='candidate' and a.onednn_receipt:
                        lib=json.loads(a.onednn_receipt.read_text());kwargs.update(expected_onednn_sha256=lib['library_sha256'],expected_onednn_version=lib['version'])
                    D.R.http(BASE,'/collective_rpc',dict(method='inventory',args=[],kwargs=kwargs))
                    for window in windows:
                        D.R.http(BASE,'/reset_prefix_cache',{})
                        phase=destination/window['name'];phase.mkdir()
                        for mode in ['cold','warm']:
                            row=W.wave(window,window['concurrency'],mode,block*2+(role=='candidate'),3,phase,512)
                            assert row['native']['preemptions']==0
                            if mode=='warm':assert row['native']['cached_tokens']>0
                            arm.append(dict(window=window['name'],mode=mode,result=row));save()
                            print('SCREEN',block,role,window['name'],mode,row['native_request_weighted_decode_tps'],row['client_aggregate_decode_tps'],flush=True)
                    if a.capacity and block==0 and role=='candidate':
                        D.R.http(BASE,'/reset_prefix_cache',{})
                        target=destination/'near-limit';target.mkdir();state['capacity']['near-limit']=D.long_case('near-limit',target,262144)
                        assert state['capacity']['near-limit']['native']['preemptions']==0
                        assert all(state['capacity']['near-limit']['usage'][k]==v for k,v in {'prompt_tokens':261120,'completion_tokens':1024,'total_tokens':262144}.items())
                        save();print('PASS_CANDIDATE_262144_TOTAL_CAPACITY',flush=True)
                    kwargs['label']='final';D.R.http(BASE,'/collective_rpc',dict(method='inventory',args=[],kwargs=kwargs))
                    worker.stop();worker=None
            state['status']='COMPLETE_PAIRED_SCREEN_REVIEW_REQUIRED'
        except BaseException as e:state.update(status='FAILED',error=repr(e));raise
        finally:
            if worker is not None:worker.stop()
            subprocess.run(['systemctl','--user','start','b70-qwen38-vllm.service'],check=True,timeout=90)
            state['restoration']=restore.wait_for_qualified_service(baseline,timeout=900)
            state['finished_unix']=time.time();save();print('RESTORED_QUALIFIED_PRODUCTION',flush=True)
if __name__=='__main__':main()

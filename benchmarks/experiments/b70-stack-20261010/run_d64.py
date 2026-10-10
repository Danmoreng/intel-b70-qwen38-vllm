#!/usr/bin/env python3
"""Two named missing-reference probes; never old-stack qualification/bisect."""
import argparse,fcntl,gzip,hashlib,importlib.util,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from exl3_candidate_worker import Worker,BASE,sha
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
D=module('d64_http',REPO/'scripts/run-exl3-contract-diagnostics.py')
restore=module('d64_restore',REPO/'scripts/run-exl3-review-micro.py')
def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate-image');p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
    baseline=json.loads((HERE/'baseline-manifest.json').read_text())['production']['image_id']
    old=json.loads((REPO/'config/releases/exl3-v2/production_image.json').read_text())['image_id']
    panel_path=REPO/'benchmarks/experiments/mtp4-pro-qualification-v031/panel.json.gz'
    panel=json.loads(gzip.decompress(panel_path.read_bytes()));window=next(w for w in panel['windows'] if w['name']=='source-64k')
    saved=REPO/'benchmark-results/b70-stack-20261010/P0-initial/worker/source-64k-control/source-64k-c1-warm/request-0.sse.jsonl'
    continuation=[]
    for line in saved.read_text().splitlines():
        for c in json.loads(line)['event'].get('choices',[]):continuation.extend(c.get('token_ids') or [])
    assert len(continuation)==256
    r=dict(status='RUNNING',old_image=old,qualified_image=baseline,candidate_image=a.candidate_image,arms={},started_unix=time.time(),
        missing_reference='Archived 64K old streams lack exact emitted IDs, same-prefix target distributions and all three MTP proposals. Existing short/32K quality and matched-state gates do not supply those 64K operands.',
        authorization_scope='Plan D64 expressly allows an old-image diagnostic only for named missing references: two frozen prefixes, output budget eight each, no throughput claim or baseline requalification.',
        continuation_source_sha256=sha(saved),panel_sha256=sha(panel_path),positions=[0,32],fresh_reconstructions_per_position=2)
    def save():(root/'campaign.json').write_text(json.dumps(r,indent=2)+'\n')
    snapshot=root/'diagnostic-sources';snapshot.mkdir();(snapshot/'d64_probe.py').write_bytes((HERE/'d64_probe.py').read_bytes())
    r['probe_source_sha256']=sha(snapshot/'d64_probe.py')
    def rpc(method,**kwargs):return D.R.http(BASE,'/collective_rpc',dict(method=method,args=[],kwargs=kwargs))
    with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);worker=None;save()
        try:
            subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
            assert subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()==['b70-chatui']
            for role,image in [('qualified',baseline),('historical',old)]+([('candidate',a.candidate_image)] if a.candidate_image else []):
                worker=Worker(image,root/role,'b70-stack-d64',env={'EXL3_SDPA_CACHE_CAPACITY':'64','VLLM_SERVER_DEV_MODE':'1','PYTHONPATH':'/experiment:/opt/b70-runtime'},binds=[(snapshot,'/experiment')])
                worker.profile['argv']+=['--worker-extension-cls','d64_probe.WorkerExtension']
                worker.start();arm=dict(identity=rpc('install_d64_probe'),windows=[]);r['arms'][role]=arm;save()
                for pos,repeat in [(p,n) for p in r['positions'] for n in range(2)]:
                    ids=window['ids']+continuation[:pos];label=f'prefix-{pos}-repeat-{repeat}'
                    D.R.http(BASE,'/reset_prefix_cache',{})
                    rpc('arm_d64_probe',label=label,prefix_tokens=len(ids),anchor=continuation[pos])
                    before=D.R.snapshot(BASE)
                    response=D.R.http(BASE,'/v1/completions',dict(model='Qwen3.8-27B',prompt=ids,
                        temperature=1.0,top_p=.95,top_k=20,seed=770000,max_tokens=8,ignore_eos=True,
                        return_token_ids=True,cache_salt=f'd64-fresh-{role}-{pos}-{repeat}'),timeout=1800)
                    _,native=D.R.wait_accounted(BASE,before,response['usage'])
                    assert response['usage']['prompt_tokens']==len(ids) and response['usage']['completion_tokens']==8
                    assert native['cached_tokens']==0 and native['preemptions']==0
                    probe=rpc('finish_d64_probe')
                    for choice in response['choices']:
                        if choice.get('prompt_token_ids') is not None:
                            assert choice.pop('prompt_token_ids')==ids
                        assert choice['token_ids'][0]==continuation[pos]
                    arm['windows'].append(dict(position=pos,repeat=repeat,prompt_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),response=response,native=native,capture=probe));save()
                    print('D64_CAPTURED',role,pos,repeat,flush=True)
                worker.stop();worker=None
            r['status']='COMPLETE_BOUNDED_D64_CAPTURE_COMPARISON_PENDING'
        except BaseException as e:r.update(status='FAILED',error=repr(e));raise
        finally:
            if worker is not None:worker.stop()
            subprocess.run(['systemctl','--user','start','b70-qwen38-vllm.service'],check=True,timeout=90)
            r['restoration']=restore.wait_for_qualified_service(baseline,timeout=900);r['finished_unix']=time.time();save()
            print('RESTORED_QUALIFIED_PRODUCTION',flush=True)
if __name__=='__main__':main()

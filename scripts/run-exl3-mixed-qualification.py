#!/usr/bin/env python3
"""Bounded ABBA mixed-serving tails, followed by final-profile API/media gates."""
import argparse
import concurrent.futures
import fcntl
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import threading
import time

from exl3_candidate_worker import Worker, REPO, BASE, sha


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,REPO/'scripts'/file)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


M=module('mixed_mtp','run-exl3-mtp-study.py')
D=module('mixed_contract','run-exl3-contract-diagnostics.py')


def mixed(panel,decoders,root,index):
    root.mkdir()
    short=next(w for w in panel['windows'] if w['name']=='code-4096')
    long=next(w for w in panel['windows'] if w['name']=='code-49152')
    def payload(w,budget,i):
        return dict(model='Qwen3.8-27B',prompt=w['ids'],temperature=0,top_p=1,top_k=-1,
                    seed=20261001,max_tokens=budget,ignore_eos=True,return_token_ids=True,
                    stream=True,stream_options={'include_usage':True},
                    cache_salt=f'mixed-final-v1-{index}-{decoders}-{i}')
    before=M.R.snapshot(BASE);started=time.monotonic();events=[threading.Event() for _ in range(decoders)]
    barrier=threading.Barrier(decoders)
    with concurrent.futures.ThreadPoolExecutor(max_workers=decoders+1) as pool:
        futures=[pool.submit(M.stream,payload(short,4096,i),barrier,root/f'decode-{i}.jsonl',events[i]) for i in range(decoders)]
        for event in events:
            if not event.wait(180):raise RuntimeError('Decode did not emit32 tokens before incoming prompt')
        assert not any(f.done() for f in futures),'Decode completed before incoming prompt'
        incoming=pool.submit(M.stream,payload(long,256,decoders),threading.Barrier(1),root/'incoming.jsonl')
        cold=incoming.result();responses=[f.result() for f in futures]
    expected=dict(prompt_tokens=decoders*4096+49152,completion_tokens=decoders*4096+256)
    deadline=time.monotonic()+45
    while True:
        native=M.R.delta(M.R.snapshot(BASE),before)
        if (native.get('completed')==decoders+1 and native.get('prompt_tokens')==expected['prompt_tokens']
                and native.get('generation_tokens')==expected['completion_tokens']):break
        if time.monotonic()>deadline:raise RuntimeError('Mixed accounting mismatch '+str(native))
        time.sleep(.25)
    assert native['completed']==decoders+1 and native['preemptions']==0 and native['cached_tokens']==0,native
    left=cold['started_monotonic'];right=cold['first_token_monotonic']
    gaps=[]
    for r in responses:
        absolute=[r['started_monotonic']+b['elapsed_s'] for b in r['bursts']]
        gaps.extend(b-a for a,b in zip(absolute,absolute[1:]) if b>=left and a<=right)
    assert gaps,'Incoming prefill did not overlap ongoing decode'
    result=dict(status='COMPLETE',decoders=decoders,total_admitted=decoders+1,
                wall_s=time.monotonic()-started,native=native,decode=responses,incoming=cold,
                incoming_prefill_client_s=right-left,
                during_incoming_prefill_burst_gaps_s=dict(count=len(gaps),p50=M.percentile(gaps,.5),
                    p95=M.percentile(gaps,.95),maximum=max(gaps)),
                note='SSE burst gaps on existing decode streams during cold49K prefill; not per-token GPU timing. ABBA changes only guarded prefill, M04 remains on.')
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return {k:v for k,v in result.items() if k not in ('decode','incoming')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True);p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=args.out.resolve();assert not root.exists()
    manifest=json.loads(args.manifest.read_text());raw=gzip.decompress(args.panel.read_bytes())
    assert sha(args.panel)==manifest['panel_gzip_sha256']
    import hashlib
    assert hashlib.sha256(raw).hexdigest()==manifest['panel_raw_sha256'];panel=json.loads(raw)
    with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);M.R.cap()
        subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True)
        root.mkdir();state=dict(status='RUNNING',started_unix=time.time(),arms=[],order=[False,True,True,False],
                               image_id=None,panel_sha256=manifest['panel_raw_sha256'],
                               source_sha256={f:sha(REPO/'scripts'/f) for f in
                                   ['run-exl3-mixed-qualification.py','exl3_candidate_worker.py','run-exl3-mtp-study.py']})
        def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
        save()
        try:
            for index,enabled in enumerate(state['order']):
                path=root/f'arm-{index}-prefill{int(enabled)}'
                worker=Worker(args.image,path,env={'EXL3_GUARDED_PREFILL':str(int(enabled))})
                state['image_id']=worker.image
                arm=dict(prefill=enabled,waves=[],mixed=[]);state['arms'].append(arm);save()
                try:
                    worker.start()
                    short=next(w for w in panel['windows'] if w['name']=='code-4096')
                    for c in [1,4]:M.wave(short,c,'warmup',index,3,path,64)
                    for n in [1,3]:
                        print('MIXED',index,'prefill',enabled,'ongoing',n,flush=True)
                        arm['mixed'].append(mixed(panel,n,path/f'mixed-c{n+1}',index));save()
                    if index==1:
                        api=path/'api';api.mkdir()
                        arm['api']=D.smoke(api,262144,{'image':32,'video':4});save()
                        media=path/'media';media.mkdir()
                        arm['media']=D.media_case(media,{'image':32,'video':4});save()
                        # Count limit AND maximum capped image area together.
                        large={'type':'image_url','image_url':{'url':D.png_url(2048,2048)}}
                        before=D.R.snapshot(BASE)
                        response=D.R.http(BASE,'/v1/chat/completions',D.chat([large]*32+[
                            {'type':'text','text':'What color are all32 pictures? Answer with one word.'}]),timeout=1800)
                        _,native=D.R.wait_accounted(BASE,before,response['usage'])
                        assert 'red' in response['choices'][0]['message']['content'].lower()
                        assert native['preemptions']==0,native
                        arm['maximum_area_images']=dict(status='PASS',usage=response['usage'],native=native,
                                                       reply=response['choices'][0]['message']['content'])
                        save()
                    arm['status']='COMPLETE';save()
                finally:worker.stop()
            state['status']='COMPLETE_MIXED_AND_API_MEDIA_GATES'
        except BaseException as exc:state['status']='FAILED';state['error']=repr(exc);raise
        finally:
            subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True)
            state['production_left_offline']=True;state['finished_unix']=time.time();save()


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Matched MTP3/MTP4 serving study; component GPU profiling is a separate pass."""
import argparse
import concurrent.futures
import fcntl
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import urllib.request

REPO=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('mtp_contract',REPO/'scripts/run-exl3-contract-diagnostics.py')
D=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(D)
R=D.R
BASE='http://127.0.0.1:8082'
NAME='b70-exl3-mtp-study'


def sha(data):return hashlib.sha256(data).hexdigest()


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def selected_cases(windows, compact=False):
    return [(w, c, mode) for w in windows for c in [1,4] for mode in ['cold','warm']
            if not compact or (w['context_tokens'] in [4096,49152]
                               or (w['name']=='code-102752' and c==1))]


def reuse_arm(source, image, panel_sha, settings, cases):
    campaign=json.loads((source/'campaign.json').read_text())
    assert campaign['status'] in ['FAILED','CURTAILED_BY_USER','COMPLETE'], 'Reuse requires a terminal campaign'
    assert campaign['image_id']==image and campaign['panel_sha256']==panel_sha, 'Reuse identity mismatch'
    arm=campaign['arms'][0]
    assert arm['depth']==3 and arm['settings']=={**settings,'speculative_config':{'method':'mtp','num_speculative_tokens':3}}, 'Reuse settings mismatch'
    rows=[]
    for window,c,mode in cases:
        key=(window['name'],c,mode)
        matches=[r for r in arm['waves'] if (r['window'],r['concurrency'],r['cache_mode'])==key]
        assert len(matches)==1, 'Missing or duplicated completed source wave: '+str(key)
        path=source/'arm-0-mtp3'/f'{key[0]}-c{c}-{mode}'/'result.json'
        row=json.loads(path.read_text());summary={k:v for k,v in row.items() if k!='responses'}
        assert summary==matches[0], 'Raw result differs from persisted wave'
        assert len(row['responses'])==c and row['native']['completed']==c
        assert all(r['prompt_ids_verified'] and len(r['token_ids'])==512 and r['usage']['prompt_tokens']==len(window['ids']) for r in row['responses'])
        rows.append({**summary,'reused_result':str(path.resolve()),'reused_result_sha256':sha(path.read_bytes())})
    return {**arm,'waves':rows,'status':'COMPLETE_REUSED_SUBSET',
            'source_campaign':str((source/'campaign.json').resolve()),
            'source_campaign_sha256':sha((source/'campaign.json').read_bytes())}


def metrics():
    with urllib.request.urlopen(BASE+'/metrics',timeout=15) as response:raw=response.read().decode()
    counters={}
    for line in raw.splitlines():
        match=re.fullmatch(r'([^\s{]+)(\{[^}]*\})?\s+([0-9.eE+-]+)',line)
        if match:
            name,labels,value=match.groups()
            if name=='vllm:spec_decode_num_accepted_tokens_per_pos_total':
                position=re.search(r'position="(\d+)"',labels or '')
                if position:name+=':position'+position.group(1)
            counters[name]=counters.get(name,0)+float(value)
    return raw,counters


def percentile(values,p):
    if not values:return None
    values=sorted(values);at=(len(values)-1)*p;i=int(at);j=min(i+1,len(values)-1)
    return values[i]+(values[j]-values[i])*(at-i)


def stream(payload,barrier,output):
    request=urllib.request.Request(BASE+'/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    barrier.wait();started=time.monotonic();first=None;usage=None;times=[];ids=[];bursts=[];chunks=[];finish=None;prompt_verified=False
    with urllib.request.urlopen(request,timeout=3600) as response,output.open('w') as log:
        for line in response:
            if not line.startswith(b'data: '):continue
            if line.strip()==b'data: [DONE]':break
            event=json.loads(line[6:]);elapsed=time.monotonic()-started
            if event.get('error'):raise RuntimeError(event['error'])
            # Preserve prompt hashes without duplicating the whole 128K prompt in every SSE log.
            for choice in event.get('choices',[]):
                returned=choice.pop('prompt_token_ids',None)
                if returned is not None:
                    assert returned==payload['prompt'],'API prompt IDs differ from the frozen inputs'
                    prompt_verified=True
                    choice['prompt_ids_sha256']=sha(json.dumps(returned,separators=(',',':')).encode())
                new=choice.get('token_ids') or []
                if new:
                    if first is None:first=elapsed
                    ids.extend(new);times.extend([elapsed]*len(new));bursts.append({'elapsed_s':elapsed,'tokens':len(new)})
                chunks.append(choice.get('text') or '');finish=choice.get('finish_reason') or finish
            usage=event.get('usage') or usage
            log.write(json.dumps({'elapsed_s':elapsed,'event':event})+'\n')
    wall=time.monotonic()-started
    assert prompt_verified,'API did not return the frozen prompt IDs for verification'
    assert usage and usage['completion_tokens']==payload['max_tokens']==len(ids)
    assert usage['prompt_tokens']==len(payload['prompt'])
    gaps=[b-a for a,b in zip(times,times[1:])]
    return {'wall_s':wall,'ttft_client_s':first,'started_monotonic':started,'finished_monotonic':started+wall,
            'first_token_monotonic':started+first if first is not None else None,'usage':usage,'finish_reason':finish,
            'prompt_ids_verified':prompt_verified,'output_ids_sha256':sha(json.dumps(ids,separators=(',',':')).encode()),
            'output_text_sha256':sha(''.join(chunks).encode()),'token_ids':ids,'bursts':bursts,
            'client_token_delivery_gaps_s':{'p50':percentile(gaps,.5),'p95':percentile(gaps,.95),'max':max(gaps,default=0)},
            'delivery_note':'Tokens in one SSE burst share the observed delivery timestamp. These are client gaps, not separate GPU-step timings. Returning prompt IDs adds identical first-packet overhead to both arms.'}


def wave(window,concurrency,mode,index,depth,phase,output_tokens):
    root=phase/f'{window["name"]}-c{concurrency}-{mode}';root.mkdir()
    payloads=[{'model':'Qwen3.8-27B','prompt':window['ids'],'temperature':0,'top_p':1,'top_k':-1,
               'seed':20261001,'max_tokens':output_tokens,'ignore_eos':True,'return_token_ids':True,
               'stream':True,'stream_options':{'include_usage':True},
               'cache_salt':f'exl3-mtp-v1-arm{index}-{window["name"]}-c{concurrency}-'+('warmup' if mode=='warmup' else 'measured')+f'-request{i}'} for i in range(concurrency)]
    # Cold and warm waves have identical salts within an arm; salts differ across requests/arms.
    before=R.snapshot(BASE);raw_before,extra_before=metrics();energy_before=R.energy();started=time.monotonic()
    barrier=threading.Barrier(concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        responses=list(pool.map(lambda pair:stream(pair[1],barrier,root/f'request-{pair[0]}.sse.jsonl'),enumerate(payloads)))
    wall=time.monotonic()-started;energy_after=R.energy();deadline=time.monotonic()+45
    while True:
        after=R.snapshot(BASE);native=R.delta(after,before)
        if (native.get('completed')==concurrency and native.get('generation_tokens')==concurrency*output_tokens
                and native.get('prompt_tokens')==concurrency*len(window['ids'])):break
        if time.monotonic()>deadline:raise RuntimeError('Wave native accounting mismatch: '+str(native))
        time.sleep(.25)
    raw_after,extra_after=metrics()
    if mode=='cold':assert native['cached_tokens']==0,'Cold wave unexpectedly reused cache'
    rounds=extra_after.get('vllm:spec_decode_num_drafts_total',0)-extra_before.get('vllm:spec_decode_num_drafts_total',0)
    if rounds<=0:raise RuntimeError('No native speculative rounds were recorded')
    positions={str(p):extra_after.get(f'vllm:spec_decode_num_accepted_tokens_per_pos_total:position{p}',0)-extra_before.get(f'vllm:spec_decode_num_accepted_tokens_per_pos_total:position{p}',0) for p in range(depth)}
    intervals=sorted((r['first_token_monotonic'],r['finished_monotonic']) for r in responses)
    union=0;left,right=intervals[0]
    for start,end in intervals[1:]:
        if start<=right:right=max(right,end)
        else:union+=right-left;left,right=start,end
    union+=right-left
    native_ttft_name='vllm:time_to_first_token_seconds_sum'
    result={'window':window['name'],'domain':window['domain'],'context_tokens':len(window['ids']),
            'concurrency':concurrency,'cache_mode':mode,'wall_s':wall,'native':native,'responses':responses,
            'speculative_rounds_sum':rounds,'accepted_tokens_by_position':positions,
            'accepted_per_round_by_position':{p:n/rounds for p,n in positions.items()},
            'generated_tokens_per_round':native['generation_tokens']/rounds,
            'mean_acceptance_length':1+native['accepted_tokens']/rounds,
            'native_prefill_tps':native['prefill_tokens']/native['prefill_seconds'] if native['prefill_seconds']>0 else None,
            'native_request_weighted_decode_tps':native['generation_tokens']/native['decode_seconds'],
            'wave_aggregate_output_tps':native['generation_tokens']/wall,
            'client_decode_interval_union_s':union,
            'client_aggregate_decode_tps':native['generation_tokens']/union,
            'native_ttft_seconds_sum':extra_after[native_ttft_name]-extra_before.get(native_ttft_name,0) if native_ttft_name in extra_after else None,
            'client_ttft_s':[r['ttft_client_s'] for r in responses],
            'energy_j':(energy_after-energy_before)/1e6 if energy_before is not None and energy_after is not None and energy_after>=energy_before else None,
            'metrics_note':'Native request decode seconds can overlap at C4; weighted decode and wave wall throughput are reported separately. Warm cache hits must be observed, not assumed.'}
    result['joules_per_output_token']=result['energy_j']/native['generation_tokens'] if result['energy_j'] is not None else None
    result['load_classification']='preempting_pressure' if native['preemptions'] else 'no_observed_preemption'
    result['comparison_note']='Preempting waves include pressure/recompute effects and cannot establish clean kernel-speed gains. Logical cache-miss prefill tokens are not an accounting of additional recomputed rows.'
    save(root/'result.json',result);(root/'metrics-before.prom').write_text(raw_before);(root/'metrics-after.prom').write_text(raw_after)
    return {k:v for k,v in result.items() if k!='responses'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True);parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--long-campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--compact',action='store_true')
    parser.add_argument('--reuse-mtp3-campaign',type=Path)
    args=parser.parse_args();root=args.output.resolve()
    if root.exists():raise RuntimeError('Fresh study directory required')
    encoded=args.panel.read_bytes();panel_bytes=gzip.decompress(encoded);panel=json.loads(panel_bytes);manifest=json.loads(args.manifest.read_text())
    if sha(panel_bytes)!=manifest['panel_raw_sha256'] or sha(encoded)!=manifest['panel_gzip_sha256']:
        raise RuntimeError('Frozen performance panel identity mismatch')
    image=subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()
    windows=[w for w in panel['windows'] if not args.pilot or w['name']=='code-4096']
    assert not (args.pilot and args.compact), 'Pilot and compact are exclusive'
    assert args.compact or not args.reuse_mtp3_campaign, 'Reuse is restricted to compact screening'
    order=[3,4] if args.pilot or args.compact else [3,4,4,3]
    cases=selected_cases(windows,args.compact)
    lockpath=REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock'
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        long=json.loads((args.long_campaign/'candidate-campaign.json').read_text())
        if long['status']!='COMPLETE' or long['image_id']!=image:
            raise RuntimeError('Matched long candidate quality must complete before performance')
        config=long['engine_config'];settings=json.loads((REPO/'config/experiments/exl3-migration/target-upstream-expanded.json').read_text())
        for key,value in settings.items():
            if config.get(key)!=value:raise RuntimeError('Serving profile differs from long quality: '+key)
        reused=reuse_arm(args.reuse_mtp3_campaign.resolve(),image,sha(panel_bytes),settings,cases) if args.reuse_mtp3_campaign else None
        root.mkdir();state={'schema':1,'status':'RUNNING','scope':'SERVING_COMPACT_SCREEN' if args.compact else ('PILOT' if args.pilot else 'SERVING_ABBA'),
            'image_id':image,'panel_sha256':sha(panel_bytes),'started_unix':time.time(),'arms':[],
            'order':order,'power_w':180,'production_restored':False,
            'source_sha256':sha(Path(__file__).read_bytes()),'long_campaign_sha256':sha((args.long_campaign/'candidate-campaign.json').read_bytes()),
            'unmeasured_required_component_metrics':['target/draft/head/sampler GPU latency','actual/padded rows and graph bucket','per-cycle GPU latency; require separate diagnostic profiling before MTP selection']}
        state['planned_waves']=len(cases)*len(order)
        state['comparison_design']='Single-pass screening; reused MTP3 is historical, no ABBA/order-drift confidence. Repeat only ambiguous cells.' if args.compact else 'Original protocol'
        def persist():save(root/'campaign.json',state)
        def interrupted(*args):raise InterruptedError('MTP study interrupted')
        signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted);persist()
        checkpoint=Path.home()/'.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw'
        try:
            subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90);R.cap()
            for index,depth in enumerate(order):
                if index==0 and reused:
                    state['arms'].append(reused);persist();continue
                path=root/f'arm-{index}-mtp{depth}';path.mkdir();phase=path
                cache=Path.home()/'.cache/exl3xpu/migration-mtp'/image.removeprefix('sha256:')/f'mtp{depth}'
                command=['docker','run','-d','--name',NAME,'--device','/dev/dri','-v','/dev/dri/by-path:/dev/dri/by-path:ro','--shm-size','8g',
                    '-p','127.0.0.1:8082:8000','-e','HF_HUB_OFFLINE=1','-e','PYTHONPATH=/opt/b70-runtime',
                    '-v',str(checkpoint)+':/models/checkpoint:ro','-v',str(REPO/'runtime')+':/opt/b70-runtime:ro']
                for key,target in [('vllm','/root/.cache/vllm'),('triton','/root/.triton/cache'),('neo_compiler_cache','/root/.cache/neo_compiler_cache')]:
                    (cache/key).mkdir(parents=True,exist_ok=True);command+=['-v',str(cache/key)+':'+target]
                command+=[image,'models/qwen3.8-27b-exl3-4.00bpw/migration-target-200704-c4.yaml','--gpu','0','--port','8000','--model-path','/models/checkpoint']
                applied={**settings,'speculative_config':{'method':'mtp','num_speculative_tokens':depth}}
                for key,value in applied.items():command+=['--set','vllm.'+key+'='+json.dumps(value,separators=(',',':'))]
                arm={'depth':depth,'command':command,'settings':applied,'waves':[]};state['arms'].append(arm);persist()
                try:
                    subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
                    deadline=time.monotonic()+900
                    while True:
                        try:R.http(BASE,'/v1/models',timeout=3);break
                        except OSError:
                            item=json.loads(subprocess.check_output(['docker','inspect',NAME],text=True))[0]
                            if not item['State']['Running']:raise RuntimeError('MTP engine exited during startup')
                            if time.monotonic()>deadline:raise RuntimeError('MTP startup timed out')
                            time.sleep(2)
                    identity=R.identity(NAME);assert identity['image_id']==image;arm['identity']=identity;persist()
                    # The warmup uses unique salts and is excluded from all measured wave summaries.
                    short=next(w for w in panel['windows'] if w['name']=='code-4096')
                    for concurrency in [1,4]:wave(short,concurrency,'warmup',index,depth,phase,panel['sampling']['max_tokens'])
                    for window,concurrency,mode in cases:
                        print(f'RUN arm{index} MTP{depth} {window["name"]} C{concurrency} {mode}',flush=True)
                        row=wave(window,concurrency,mode,index,depth,phase,panel['sampling']['max_tokens'])
                        arm['waves'].append(row);persist()
                    arm['status']='COMPLETE';persist()
                finally:
                    logs=subprocess.run(['docker','logs',NAME],capture_output=True,text=True)
                    (path/'worker.log').write_text(logs.stdout+logs.stderr)
                    subprocess.run(['docker','stop','-t','20',NAME],capture_output=True,timeout=60)
                    subprocess.run(['docker','rm','-f',NAME],capture_output=True,timeout=30)
            state['status']='COMPLETE'
        except BaseException as exc:
            state['status']='FAILED';state['error']=repr(exc);raise
        finally:
            subprocess.run(['docker','rm','-f',NAME],capture_output=True,timeout=30)
            subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
            state['production_left_offline']=True;state['finished_unix']=time.time();persist()


if __name__=='__main__':main()

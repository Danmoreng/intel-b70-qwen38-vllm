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

REPO=Path(__file__).resolve().parents[3]
SPEC=importlib.util.spec_from_file_location('mtp_contract',REPO/'scripts/run-exl3-contract-diagnostics.py')
D=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(D)
R=D.R
BASE='http://127.0.0.1:8082'
NAME='b70-exl3-mtp-study'


def sha(data):return hashlib.sha256(data).hexdigest()


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


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

def stream(payload,barrier,output,first_tokens_event=None):
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
                    if first_tokens_event is not None and len(ids)>=32:first_tokens_event.set()
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
    payloads=[{'model':'Qwen3.8-27B','prompt':window['ids'],'temperature':1.0,'top_p':.95,'top_k':20,
               'seed':770000+i,'max_tokens':output_tokens,'ignore_eos':True,'return_token_ids':True,
               'stream':True,'stream_options':{'include_usage':True},
               'cache_salt':f'exl3-mtp-v1-arm{index}-{window["name"]}-c{concurrency}-'+('warmup' if mode=='warmup' else 'measured')+f'-request{i}'} for i in range(concurrency)]
    # Cold and warm waves have identical salts within an arm; salts differ across requests/arms.
    before=R.snapshot(BASE);raw_before,extra_before=metrics();energy_before=None;started=time.monotonic()
    barrier=threading.Barrier(concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        responses=list(pool.map(lambda pair:stream(pair[1],barrier,root/f'request-{pair[0]}.sse.jsonl'),enumerate(payloads)))
    wall=time.monotonic()-started;energy_after=None;deadline=time.monotonic()+45
    while True:
        after=R.snapshot(BASE);native=R.delta(after,before)
        if (native.get('completed')==concurrency and native.get('generation_tokens')==concurrency*output_tokens
                and native.get('prompt_tokens')==concurrency*len(window['ids'])):break
        if time.monotonic()>deadline:raise RuntimeError('Wave native accounting mismatch: '+str(native))
        time.sleep(.25)
    raw_after,extra_after=metrics()
    computed_name='vllm:request_prefill_kv_computed_tokens_sum'
    assert computed_name in extra_after and computed_name in extra_before
    computed=extra_after[computed_name]-extra_before[computed_name]
    recompute_excess=computed+native['cached_tokens']-native['prompt_tokens']
    assert recompute_excess==0,('Unexpected prefill accounting',computed,native)
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
    decode_tokens=native['generation_tokens']-native['completed']
    result={'window':window['name'],'domain':window['domain'],'context_tokens':len(window['ids']),
            'concurrency':concurrency,'cache_mode':mode,'wall_s':wall,'native':native,'responses':responses,
            'speculative_rounds_sum':rounds,'accepted_tokens_by_position':positions,
            'accepted_per_round_by_position':{p:n/rounds for p,n in positions.items()},
            'generated_tokens_per_round':native['generation_tokens']/rounds,
            'mean_acceptance_length':1+native['accepted_tokens']/rounds,
            'native_prefill_tps':native['prefill_tokens']/native['prefill_seconds'] if native['prefill_seconds']>0 else None,
            'post_first_completion_tokens':decode_tokens,
            'prefill_computed_tokens':computed,'prefill_recompute_excess':recompute_excess,
            'native_request_weighted_decode_tps':decode_tokens/native['decode_seconds'],
            'wave_aggregate_output_tps':native['generation_tokens']/wall,
            'client_decode_interval_union_s':union,
            'client_aggregate_decode_tps':decode_tokens/union,
            'native_ttft_seconds_sum':extra_after[native_ttft_name]-extra_before.get(native_ttft_name,0) if native_ttft_name in extra_after else None,
            'client_ttft_s':[r['ttft_client_s'] for r in responses],
            'energy_j':(energy_after-energy_before)/1e6 if energy_before is not None and energy_after is not None and energy_after>=energy_before else None,
            'metrics_note':'Native request decode seconds can overlap at C4; weighted decode and wave wall throughput are reported separately. Warm cache hits must be observed, not assumed.'}
    result['joules_per_output_token']=result['energy_j']/native['generation_tokens'] if result['energy_j'] is not None else None
    result['load_classification']='preempting_pressure' if native['preemptions'] else 'no_observed_preemption'
    result['comparison_note']='Preempting waves include pressure/recompute effects and cannot establish clean kernel-speed gains. Logical cache-miss prefill tokens are not an accounting of additional recomputed rows.'
    save(root/'result.json',result);(root/'metrics-before.prom').write_text(raw_before);(root/'metrics-after.prom').write_text(raw_after)
    return {k:v for k,v in result.items() if k!='responses'}

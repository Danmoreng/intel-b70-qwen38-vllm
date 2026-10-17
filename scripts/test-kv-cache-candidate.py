#!/usr/bin/env python3
"""Real API tests: cold/warm CPU cache reuse and memory-aware admission, synthetic input only."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import urllib.request


def post(base, endpoint, value, timeout=900):
    req=urllib.request.Request(base+endpoint, data=json.dumps(value).encode(), headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response: return json.load(response)


def metrics(base):
    with urllib.request.urlopen(base+'/metrics',timeout=10) as response: raw=response.read().decode()
    result={}
    for line in raw.splitlines():
        m=re.fullmatch(r'([^#{ ]+)(?:\{([^}]*)\})?\s+([0-9.eE+-]+)',line)
        if m:
            result[m[1]]=result.get(m[1],0)+float(m[3])
            # Preserve direction for actual RAM -> GPU restoration evidence.
            direction=re.search(r'(?:^|,)transfer_type="([^"]+)"',m[2] or '')
            if direction:
                key=m[1]+':'+direction[1]
                result[key]=result.get(key,0)+float(m[3])
    return result


def prompt(base, target, key):
    # Different nonce at the beginning deliberately prevents shared-prefix hits.
    text=f'Cache test {key}. Remember the marker ORANGE-{key}.\n'+('The river flows past the quiet village.\n'*(target//8+64))
    count=post(base,'/tokenize',{'model':'Qwen3.8-27B','prompt':text})['count']
    words=text.splitlines(); words=words[:max(2,int(len(words)*target/count))]
    if key.startswith('retention-'):
        # The answer must depend on cached history, not on an answer repeated
        # in the final uncached suffix.
        codes=needles(key)
        for position,code in zip([2,len(words)//2,len(words)*3//4],codes):
            words[position]=f'The secret record is {code}. Remember it exactly.'
        text='\n'.join(words)+'\nReturn the three secret records in their original order, separated by spaces. Nothing else.'
    else:
        text='\n'.join(words)+f'\nAnswer only: ORANGE-{key}. Do not explain.'
    return [{'role':'user','content':text}]


def needles(key):
    return [label+'-'+hashlib.sha256((key+label).encode()).hexdigest()[:8]
            for label in ['ALPHA','BETA','GAMMA']]


def request(base, messages, output=64, force_output=False, minimum_cached_tokens=0):
    payload={'model':'Qwen3.8-27B','messages':messages,'temperature':0,'seed':42,
        'max_tokens':output,'stream':True,'stream_options':{'include_usage':True},
        'chat_template_kwargs':{'enable_thinking':False}}
    if force_output:payload['ignore_eos']=True
    req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    previous_cache=metrics(base) if minimum_cached_tokens else None
    cache_at_admission=None
    start=time.monotonic();first=None;text=[];usage=None
    with urllib.request.urlopen(req,timeout=1200) as response:
        if minimum_cached_tokens:
            deadline=time.monotonic()+15
            while True:
                current=metrics(base)
                queried=current.get('vllm:prefix_cache_queries_total',0)>previous_cache.get('vllm:prefix_cache_queries_total',0)
                local_hit=current.get('vllm:prefix_cache_hits_total',0)-previous_cache.get('vllm:prefix_cache_hits_total',0)
                external_queried=current.get('vllm:external_prefix_cache_queries_total',0)>previous_cache.get('vllm:external_prefix_cache_queries_total',0)
                if queried and (local_hit>=minimum_cached_tokens or (
                        external_queried and current.get('vllm:num_requests_running',0)>0)):
                    cache_at_admission={
                        'gpuTokens':current.get('vllm:prefix_cache_hits_total',0)-previous_cache.get('vllm:prefix_cache_hits_total',0),
                        'cpuTokens':current.get('vllm:external_prefix_cache_hits_total',0)-previous_cache.get('vllm:external_prefix_cache_hits_total',0)}
                    if sum(cache_at_admission.values())<minimum_cached_tokens:
                        raise RuntimeError(f'Insufficient cached prefix at admission: {cache_at_admission}; request cancelled before full recomputation')
                    break
                if time.monotonic()>deadline:raise RuntimeError('Cache-admission metrics did not arrive')
                time.sleep(.1)
        for line in response:
            if not line.startswith(b'data: ') or line.strip()==b'data: [DONE]':continue
            event=json.loads(line[6:])
            if event.get('error'):raise RuntimeError(event['error'])
            if event.get('usage'):usage=event['usage']
            for choice in event.get('choices',[]):
                content=choice.get('delta',{}).get('content') or choice.get('delta',{}).get('reasoning_content') or ''
                if content and first is None:first=time.monotonic()
                text.append(content)
    result={'seconds':time.monotonic()-start,'ttftSeconds':None if first is None else first-start,
        'usage':usage,'outputSha256':hashlib.sha256(''.join(text).encode()).hexdigest(),'output':''.join(text)}
    if cache_at_admission is not None:result['cacheAtAdmission']=cache_at_admission
    print(json.dumps({'requestCompleted':{k:result[k] for k in ['seconds','ttftSeconds','usage']}}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8082')
    p.add_argument('--mode',choices=['smoke','retention','admission','parallel','cancellation','rotation','parallel-restore','capacity','two-full'],default='smoke')
    p.add_argument('--tokens',type=int,default=16000);p.add_argument('--count',type=int,default=3)
    p.add_argument('--secondary-tokens',type=int)
    p.add_argument('--out',type=Path,default=Path('benchmark-results/kv-ram-candidate/results.json'))
    a=p.parse_args();before=metrics(a.base);rows=[];samples=[];done=threading.Event();result=None
    if a.mode=='two-full' and a.count!=2:p.error('two-full requires --count 2')
    def observe():
        while not done.wait(1):
            try:
                m=metrics(a.base)
                available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
                samples.append({'at':time.time(),'hostMemAvailableBytes':available,
                    **{k:v for k,v in m.items() if any(s in k for s in ['num_requests','num_preemptions','kv_cache_usage','kv_offload','cpu_cache'])}})
            except OSError:pass
    thread=threading.Thread(target=observe,daemon=True);thread.start()
    keys=[f'{"retention" if a.mode in ["parallel-restore","capacity"] else a.mode}-{i}' for i in range(a.count)]
    if a.mode=='two-full':keys=['retention-0','retention-full-other']
    targets=[a.tokens if i==0 or a.secondary_tokens is None else a.secondary_tokens for i in range(a.count)]
    prompts=[prompt(a.base,target,key) for target,key in zip(targets,keys)]
    resume_before=None
    capacity_resumes=[]
    try:
        if a.mode=='two-full':
            for messages in prompts:rows.append(request(a.base,messages))
            resume_before=metrics(a.base)
            for messages in prompts:
                rows.append(request(a.base,messages,minimum_cached_tokens=a.tokens-3200))
        elif a.mode=='capacity':
            for messages in prompts:
                for _ in range(3):rows.append(request(a.base,messages))
            with urllib.request.urlopen(urllib.request.Request(
                    a.base+'/reset_prefix_cache?reset_connector=false',data=b'',method='POST'),timeout=30) as response:
                if response.status!=200:raise RuntimeError('GPU cache reset failed')
            resume_before=metrics(a.base)
            for messages in prompts:
                previous=metrics(a.base)
                rows.append(request(a.base,messages))
                current=metrics(a.base)
                capacity_resumes.append({'cpuTokens':current.get('vllm:external_prefix_cache_hits_total',0)-previous.get('vllm:external_prefix_cache_hits_total',0),
                    'gpuTokens':current.get('vllm:prefix_cache_hits_total',0)-previous.get('vllm:prefix_cache_hits_total',0),
                    'cpuLoadBytes':current.get('vllm:kv_offload_total_bytes_total:CPU_to_GPU',0)-previous.get('vllm:kv_offload_total_bytes_total:CPU_to_GPU',0)})
        elif a.mode=='parallel-restore':
            rows=[request(a.base,messages) for messages in prompts]
            with urllib.request.urlopen(urllib.request.Request(
                    a.base+'/reset_prefix_cache?reset_connector=false',data=b'',method='POST'),timeout=30) as response:
                if response.status!=200:raise RuntimeError('GPU cache reset failed')
            resume_before=metrics(a.base)
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.count) as pool:
                rows.extend(pool.map(lambda messages:request(a.base,messages),prompts))
        elif a.mode=='rotation':
            favored=prompt(a.base,16000,'retention-rotation')
            rows=[request(a.base,favored),request(a.base,favored)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.count) as pool:
                rows.extend(pool.map(lambda messages:request(a.base,messages,256,True),prompts))
            resume_before=metrics(a.base)
            rows.append(request(a.base,favored))
        elif a.mode=='cancellation':
            payload={'model':'Qwen3.8-27B','messages':prompts[0],
                     'max_tokens':256,'stream':True,
                     'chat_template_kwargs':{'enable_thinking':False}}
            req=urllib.request.Request(a.base+'/v1/chat/completions',
                data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=30):
                deadline=time.monotonic()+15
                while metrics(a.base).get('vllm:num_requests_running',0)==0:
                    if time.monotonic()>deadline:raise RuntimeError('Cancellation target never started')
                    time.sleep(.1)
            deadline=time.monotonic()+30
            while True:
                m=metrics(a.base)
                if m.get('vllm:num_requests_running',0)==0 and m.get('vllm:num_requests_waiting',0)==0:break
                if time.monotonic()>deadline:raise RuntimeError('Cancelled request retained scheduler capacity')
                time.sleep(.1)
            rows.append(request(a.base,prompt(a.base,8000,'cancel-recovery')))
        elif a.mode in ['parallel','admission']:
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.count) as pool:
                rows=list(pool.map(lambda messages:request(a.base,messages,256,True),prompts))
        else:
            for messages in prompts:rows.append(request(a.base,messages))
            if a.mode=='retention':
                # Deterministically evict GPU prefixes without resetting the CPU
                # connector. Only use on this otherwise idle test server.
                with urllib.request.urlopen(urllib.request.Request(
                        a.base+'/reset_prefix_cache?reset_connector=false',
                        data=b'',method='POST'),timeout=30) as response:
                    if response.status!=200:raise RuntimeError('GPU cache reset failed')
            rows.append(request(a.base,prompts[0]))
        after=metrics(a.base)
        result={'mode':a.mode,'requestedTokens':a.tokens,'rows':rows,'samples':samples,
            'metricDelta':{k:v-before.get(k,0) for k,v in after.items() if any(s in k for s in ['preempt','kv_offload','cpu_cache','prefix_cache','prefill_kv_computed'])}}
        if resume_before is not None:
            result['resumeMetricDelta']={k:v-resume_before.get(k,0) for k,v in after.items()
                if any(s in k for s in ['kv_offload','prefix_cache','prefill_kv_computed'])}
        if capacity_resumes:result['capacityResumes']=capacity_resumes
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2),flush=True)
        if any(not row['usage'] or not row['output'].strip() for row in rows):raise RuntimeError('Incomplete response')
        if a.mode in ['smoke','retention'] and rows[0]['outputSha256']!=rows[-1]['outputSha256']:
            raise RuntimeError('Restored-prefix output differs from original deterministic response')
        if a.mode=='retention' and result['metricDelta'].get('vllm:external_prefix_cache_hits_total',0)<=0:
            raise RuntimeError('No CPU prefix reload occurred; GPU-only hit is insufficient')
        if a.mode=='retention' and result['metricDelta'].get('vllm:kv_offload_total_bytes_total:CPU_to_GPU',0)<=0:
            raise RuntimeError('No completed CPU-to-GPU transfer measured')
        if a.mode=='retention' and any(code not in row['output'] for row in [rows[0],rows[-1]]
                for code in needles('retention-0')):
            raise RuntimeError('Cached-history retrieval returned incorrect secret records')
        if a.mode=='rotation':
            if rows[0]['outputSha256']!=rows[-1]['outputSha256']:
                raise RuntimeError('History changed after natural GPU eviction and RAM reload')
            if any(code not in rows[-1]['output'] for code in needles('retention-rotation')):
                raise RuntimeError('History retrieval lost secret records after rotation')
            if result['resumeMetricDelta'].get('vllm:external_prefix_cache_hits_total',0)<=0 or result['resumeMetricDelta'].get('vllm:kv_offload_total_bytes_total:CPU_to_GPU',0)<=0:
                raise RuntimeError('No real CPU reload after natural GPU eviction')
        if a.mode=='two-full':
            for i,key in enumerate(keys):
                if rows[i]['outputSha256']!=rows[2+i]['outputSha256'] or any(
                        code not in rows[2+i]['output'] for code in needles(key)):
                    raise RuntimeError('Full history changed after the other full history ran')
            delta=result['resumeMetricDelta']
            if delta.get('vllm:external_prefix_cache_hits_total',0)<a.tokens-3200 or delta.get('vllm:kv_offload_total_bytes_total:CPU_to_GPU',0)<=0:
                raise RuntimeError('Older full history was not retained in RAM')
            if any(row['usage']['prompt_tokens']<a.tokens for row in rows):
                raise RuntimeError('Two-full test did not use full-sized prompts')
            if result['metricDelta'].get('vllm:num_preemptions_total',0)!=0:
                raise RuntimeError('Full histories triggered a preemption')
        if a.mode=='parallel-restore':
            for i,key in enumerate(keys):
                if rows[i]['outputSha256']!=rows[a.count+i]['outputSha256'] or any(
                        code not in rows[a.count+i]['output'] for code in needles(key)):
                    raise RuntimeError('Concurrent restored histories differ or contain incorrect records')
            delta=result['resumeMetricDelta']
            if delta.get('vllm:external_prefix_cache_hits_total',0)<a.count*(a.tokens-3200) or delta.get('vllm:kv_offload_total_bytes_total:CPU_to_GPU',0)<=0:
                raise RuntimeError('Not all concurrent histories were restored from CPU cache')
        if a.mode=='capacity':
            for i,key in enumerate(keys):
                if any(rows[i*3]['outputSha256']!=rows[index]['outputSha256'] for index in [i*3+1,i*3+2,a.count*3+i]) or any(
                        code not in rows[a.count*3+i]['output'] for code in needles(key)):
                    raise RuntimeError('Capacity test changed a restored history')
                if capacity_resumes[i]['cpuTokens']<targets[i]-3200 or capacity_resumes[i]['gpuTokens']!=0 or capacity_resumes[i]['cpuLoadBytes']<=0:
                    raise RuntimeError('Capacity test did not restore each full prefix from RAM')
            if sum(x['cpuTokens'] for x in capacity_resumes)<290000:
                raise RuntimeError('Less than 290000 distinct tokens retained in CPU cache')
        if a.mode in ['admission','rotation']:
            if not any(s.get('vllm:num_requests_waiting',0)>0 for s in samples):raise RuntimeError('No waiting observed')
            if any(s.get('vllm:num_requests_running',0)>1 for s in samples):raise RuntimeError('Oversized requests overlapped')
            if after.get('vllm:num_preemptions_total',0)!=before.get('vllm:num_preemptions_total',0):raise RuntimeError('Preemptions observed')
        result['validation']='passed'
        a.out.write_text(json.dumps(result,indent=2)+'\n')
    except Exception as error:
        if result is None:result={'mode':a.mode,'requestedTokens':a.tokens,'rows':rows,'samples':samples}
        result.update(validation='failed',error=f'{type(error).__name__}: {error}')
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2)+'\n')
        raise
    finally:done.set();thread.join(timeout=5)


if __name__=='__main__':main()

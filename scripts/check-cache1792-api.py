"""Functional gate for the release's real API and partial-page cache forks."""
import argparse,datetime as dt,json,re,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading,time
from zoneinfo import ZoneInfo

def post(base,path,body):
    req=urllib.request.Request(base+path,json.dumps(body).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=240) as f:return json.load(f)

def metrics(base):
    with urllib.request.urlopen(base+'/metrics',timeout=10) as f:return f.read().decode()

def run(base,root,fixture):
    root.mkdir(exist_ok=False,parents=True)
    now=dt.datetime.now(ZoneInfo('Europe/Berlin'));assert now.hour<22
    case=json.loads(fixture.read_text())['cases']['canary']
    source=case['prompt_token_ids'];assert len(source)==16384
    salt='release1792-'+str(time.time_ns());case['cache_salt']=salt
    before=metrics(base);(root/'before.prom').write_text(before)
    assert 'block_size="1792"' in before and 'prefix_match_unit="128"' in before
    def request(label,ids,expected,barrier=None,partial=False,tokens=128):
        body=dict(model='Qwen3.8-27B',prompt=ids,max_tokens=tokens,temperature=0,seed=20261112,cache_salt=salt,return_token_ids=True)
        if barrier:body['ignore_eos']=True
        (root/(label+'-request.json')).write_text(json.dumps(body)+'\n')
        if barrier:barrier.wait(timeout=15)
        reply=post(base,'/v1/completions',body);(root/(label+'-response.json')).write_text(json.dumps(reply)+'\n')
        choice=reply['choices'][0];assert choice['prompt_token_ids']==ids
        text=choice['text'];parsed=json.JSONDecoder().raw_decode(text[text.index('{'):])[0];assert parsed==expected,(label,parsed,expected)
        hit=reply['usage']['prompt_tokens_details']['cached_tokens']
        if partial:assert hit==16256 and hit%1792!=0,(label,hit)
        return dict(label=label,parsed=parsed,cached_tokens=hit,partial_page_hit=bool(hit%1792))
    rows=[request('parent-prime',source,case['expected'])]
    def tokens(text):return post(base,'/tokenize',dict(model='Qwen3.8-27B',prompt=text,add_special_tokens=False))['tokens']
    marker=tokens('<|im_start|>');assert len(marker)==1
    start=max(i for i,t in enumerate(source) if t==marker[0]);header=source[start:];assert len(header)<128
    barrier=threading.Barrier(2)
    with ThreadPoolExecutor(2) as pool:
        jobs=[]
        for label,question,keys in [('audit','Return only the stored audit value as a JSON object with exactly the key audit.',['audit']),('sides','Return only the stored left and right values as a JSON object with exactly the keys left and right.',['left','right'])]:
            ids=source+tokens('OK<|im_end|>\n<|im_start|>user\n'+question+'<|im_end|>\n')+header
            jobs.append(pool.submit(request,'child-'+label,ids,{k:case['expected'][k] for k in keys},barrier,True,256))
        rows.extend(f.result() for f in jobs)
    rows.append(request('parent-after-children',source,case['expected']))
    after=metrics(base);(root/'after.prom').write_text(after)
    def d2h(raw):return sum(float(s.rsplit(' ',1)[1]) for s in raw.splitlines() if s.startswith('vllm:kv_offload_total_bytes_total{') and 'transfer_type="GPU_to_CPU"' in s)
    transferred=d2h(after)-d2h(before);assert transferred>0
    result=dict(status='PASS_RELEASE_FACTS_PARTIAL_COW_C2_PARENT_RECHECK_AND_D2H',rows=rows,gpu_to_cpu_bytes=transferred,scope='Real release API, no developer endpoint/audit/determinism hooks. Two concurrent256-token continuations with actual16256-token partial-page hits, parent facts rechecked. Finite functional deployment gate; prior numerical proof separate.')
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.base,a.out,a.fixture)))

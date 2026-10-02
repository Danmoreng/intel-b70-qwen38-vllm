"""Bounded diverse exact-K cache soak; allocations separated from reservations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from exl3xpu.ops import _get_esimd

def memory():
    return dict(rss_bytes=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),
                allocated_bytes=torch.xpu.memory_allocated(),reserved_bytes=torch.xpu.memory_reserved())

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists();a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(20261002);op=_get_esimd();assert op is not None
    result=dict(status='RUNNING',capacity=int(os.environ.get('EXL3_SDPA_CACHE_CAPACITY','64')),
                exact_lengths=[4096]+[4097+37*i for i in range(80)],epochs=[],numerical=[],
                tolerances=dict(rtol=.01,atol=.003),
                library_sha256=hashlib.sha256(Path('/opt/exl3xpu/exl3xpu/_C.so').read_bytes()).hexdigest())
    def save():a.out.write_text(json.dumps(result,indent=2)+'\n')
    q=torch.randn((6,256,256),dtype=torch.float16,device='xpu')
    k=torch.randn((1,8192,256),dtype=torch.float16,device='xpu');v=torch.randn_like(k)
    out=torch.empty_like(q);none=torch.empty(0,device='xpu');save()
    try:
        for epoch in range(3):
            samples=[]
            for i,length in enumerate(result['exact_lengths']):
                for current in [length,length,4096]:
                    op.exl3_sdpa(q,k[:,:current],v[:,:current],out,none,True,.0625)
                    stat=dict(op.exl3_sdpa_cache_stats(q))
                    assert stat['capacity']==result['capacity'] and stat['entries']<=result['capacity']
                    samples.append(dict(length=current,cache=stat,**memory()))
                if i in (0,40,80) and epoch==0:
                    # Last operation is the repeated hot 4096-key shape.
                    scores=q[:,-64:].float()@k[0,:4096].float().T*.0625
                    visible=4096-64+torch.arange(64,device='xpu')[:,None]
                    scores.masked_fill_(torch.arange(4096,device='xpu')[None]>visible,-float('inf'))
                    gold=scores.softmax(-1)@v[0,:4096].float()
                    actual=out[:,-64:].float()
                    ok=bool(torch.allclose(actual,gold,rtol=.01,atol=.003))
                    result['numerical'].append(dict(index=i,allclose=ok,max_abs=float((actual-gold).abs().max())))
                    assert ok
                if i%20==0:print('CACHE_SOAK',epoch,i,stat,flush=True)
            torch.xpu.synchronize()
            result['epochs'].append(dict(epoch=epoch,samples=samples,final_cache=dict(op.exl3_sdpa_cache_stats(q)),**memory()));save()
        final=result['epochs'][-1]['final_cache']
        assert final['peak_entries']<=result['capacity'] and final['evictions']>0 and final['hits']>0
        first,last=result['epochs'][1],result['epochs'][2]
        result['post_warmup_growth_bytes']={key:last[key]-first[key] for key in ['rss_bytes','allocated_bytes','reserved_bytes']}
        # Prespecified generous plateau bound for this small finite panel, not
        # a universal worker-memory guarantee or a bound on oneDNN internals.
        assert all(v<=64*1024*1024 for v in result['post_warmup_growth_bytes'].values()),result['post_warmup_growth_bytes']
        result['status']='PASS_BOUNDED_APPLICATION_CACHE_FINITE_SOAK'
    except BaseException as exc:result['status']='FAILED';result['error']=repr(exc);raise
    finally:save()
    print(json.dumps(dict(status=result['status'],final_cache=final,growth=result['post_warmup_growth_bytes'])),flush=True)

if __name__=='__main__':main()

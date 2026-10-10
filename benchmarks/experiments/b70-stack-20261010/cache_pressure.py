"""Finite 81-key eviction with deliberately outstanding work on the same queue."""
import argparse,json,time
from pathlib import Path
import torch
from exl3xpu.ops import _get_esimd
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists()
    torch.manual_seed(881102);op=_get_esimd();q=torch.randn(6,256,256,device='xpu',dtype=torch.float16)
    k=torch.randn(1,8192,256,device='xpu',dtype=torch.float16);v=torch.randn_like(k);out=torch.empty_like(q);empty=torch.empty(0,device='xpu')
    keys=[4096]+[4097+37*i for i in range(80)]
    def execute(length):op.exl3_sdpa(q,k[:,:length],v[:,:length],out,empty,True,.0625)
    for length in keys:execute(length)
    torch.xpu.synchronize();before=dict(op.exl3_sdpa_cache_stats(q))
    assert before['capacity']==64 and before['entries']==64
    x=torch.randn(8192,4096,device='xpu',dtype=torch.float16);w=torch.randn(4096,4096,device='xpu',dtype=torch.float16);y=torch.empty_like(x)
    torch.mm(x,w,out=y);torch.xpu.synchronize()
    # Keep earlier completed partitions eligible at first. Cycling through all
    # 81 keys while the queue is busy makes replacement entries in flight;
    # a subsequent full-cache miss can then require victim completion.
    for _ in range(192):torch.mm(x,w,out=y)
    started=time.monotonic()
    for length in keys:execute(length)
    torch.xpu.synchronize();after=dict(op.exl3_sdpa_cache_stats(q))
    length=keys[-1];scores=q[:,-64:].float()@k[0,:length].float().T*.0625
    visible=length-64+torch.arange(64,device='xpu')[:,None]
    scores.masked_fill_(torch.arange(length,device='xpu')[None]>visible,-float('inf'))
    gold=scores.softmax(-1)@v[0,:length].float();actual=out[:,-64:].float()
    numerical=bool(torch.allclose(actual,gold,rtol=.01,atol=.003));assert numerical
    assert after['capacity']==after['entries']==after['peak_entries']==64
    assert after['evictions']>before['evictions']
    waits=after['pressure_waits']-before['pressure_waits']
    report=dict(status='PASS_INFLIGHT_CACHE_PRESSURE_REFERENCE' if waits>0 else 'PASS_EVICTION_REFERENCE_PRESSURE_BRANCH_NOT_OBSERVED',
        before=before,after=after,pressure_waits_delta=waits,numerical_allclose=numerical,max_abs=float((actual-gold).abs().max()),
        rtol=.01,atol=.003,exact_keys=keys,queued_mm_calls=192,enqueue_and_completion_seconds=time.monotonic()-started,
        scope='Artificial queue backlog is correctness diagnostics only, never serving performance or realistic load.')
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()

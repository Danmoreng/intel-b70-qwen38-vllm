"""Actual old/new M04 output copies on equal inputs; caller-owned graphs and ABBA."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import torch
from exl3xpu import shared_kv_verify as candidate
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func as native

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rows',type=int,nargs='+',choices=[2,3,4,5],default=[2,3,4,5])
    p.add_argument('--rounds',type=int,default=8);p.add_argument('--iterations',type=int,default=10);a=p.parse_args()
    assert 1<=a.rounds<=32 and 1<=a.iterations<=100
    assert not a.out.exists();a.out.parent.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('qualified_m04',a.baseline);baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    torch.ops.load_library('/opt/exl3xpu/m04/m04.so');torch.manual_seed(20261002)
    result=dict(status='RUNNING',cases=[],tolerance=dict(rtol=.01,atol=.003),abba_rounds=a.rounds,cycles_per_event=a.iterations,
        scope='Equal query/KV histories; paired micro/graph cycle latency, not full serving throughput')
    def save():a.out.write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        for batch in (1,4):
            for rows in a.rows:
                for length in (4097,102753):
                    page=1600;pages=(length+page-1)//page
                    storage=torch.randn((batch*pages,page,4,2,256),dtype=torch.float16,device='xpu').to(torch.float8_e4m3fn)
                    q=torch.randn((batch*rows,24,256),dtype=torch.float16,device='xpu');out=torch.empty_like(q)
                    d=dict(q=q,k=storage[:,:,:,0],v=storage[:,:,:,1],out=out,
                        block_table=torch.randperm(batch*pages,device='xpu').view(batch,pages).int(),
                        cu_seqlens_q=torch.arange(batch+1,device='xpu',dtype=torch.int32)*rows,
                        seqused_k=torch.full((batch,),length,device='xpu',dtype=torch.int32),max_seqlen_q=rows,max_seqlen_k=length,
                        softmax_scale=.0625,causal=True,k_descale=torch.tensor(.75,device='xpu').expand(batch,4),
                        v_descale=torch.tensor(1.25,device='xpu').expand(batch,4))
                    assert candidate.eligible(d)
                    old=baseline.run(d).clone();new=candidate.run(d).clone();assert torch.equal(old,new)
                    gold=native(**dict(d,out=torch.empty_like(out)))
                    assert torch.allclose(new.float(),gold.float(),rtol=.01,atol=.003)
                    fallback=candidate.run(dict(d,out=None));assert torch.equal(new,fallback)
                    graphs={};pointer=out.data_ptr()
                    for name,module in [('baseline',baseline),('candidate',candidate)]:
                        for _ in range(3):module.run(d)
                        torch.xpu.synchronize();graph=torch.xpu.XPUGraph()
                        with torch.xpu.graph(graph):value=module.run(d)
                        assert value is out and value.data_ptr()==pointer;graphs[name]=graph
                    graphs['baseline'].replay();before=out.clone();graphs['candidate'].replay();assert torch.equal(before,out)
                    q.mul_(.5);d['block_table'].copy_(d['block_table'].flip(1));d['seqused_k'].sub_(1)
                    graphs['baseline'].replay();changed=out.clone();graphs['candidate'].replay();assert torch.equal(changed,out)
                    assert out.data_ptr()==pointer and not torch.equal(before,changed)
                    samples={'baseline':[],'candidate':[]}
                    # Eight predetermined alternating ABBA rounds, ten graph
                    # cycles/event. Sync only at timing boundaries, not in run().
                    for _ in range(a.rounds):
                        for name in ['baseline','candidate','candidate','baseline']:
                            start,end=torch.xpu.Event(enable_timing=True),torch.xpu.Event(enable_timing=True)
                            start.record()
                            for __ in range(a.iterations):graphs[name].replay()
                            end.record();end.synchronize();samples[name].append(start.elapsed_time(end)/a.iterations)
                    med={name:statistics.median(v) for name,v in samples.items()}
                    result['cases'].append(dict(batch=batch,rows=rows,kv=length,bitexact_old_new=True,
                        native_allclose=True,fallback_bitexact=True,graph_mutation_bitexact=True,
                        output_storage_unchanged=True,median_ms=med,speedup_pct=100*(med['baseline']/med['candidate']-1),samples_ms=samples))
                    save();print('COPY_CASE',batch,rows,length,med,flush=True)
                    del graphs,d,storage,q,out,old,new,gold,fallback,before,changed
        result['status']='PASS_COPY_NUMERICAL_AND_MUTABLE_GRAPH_GATES_MICRO_LATENCY_ONLY'
    except BaseException as exc:result['status']='FAILED';result['error']=repr(exc);raise
    finally:save()

if __name__=='__main__':main()

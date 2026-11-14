"""Real checkpoint fused/head row, padding, large-first and graph regression gate.

Small-M compares with independent FP32 Hadamard/matmul. Large INT8 compares
with independently quantized torch._int_mm and the already gated transforms.
The existing 2e-3 relative-norm tolerance is fixed before this matrix.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from exl3xpu import ops, ref

ROWS = [1,2,3,4,5,8,12,16,20,24,32,40,48,64,128,129,256,512]
TOLERANCE = 2e-3


def check(actual,expected):
    a,b=actual.float(),expected.float()
    relative=float((a-b).norm()/b.norm().clamp_min(1e-20))
    finite=bool(a.isfinite().all())
    assert finite and relative<TOLERANCE,(relative,finite)
    return dict(relative_norm=relative,maximum_absolute=float((a-b).abs().max()),finite=finite)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--groups',nargs='+',help='Only newly required shape classes; prior terminal evidence stays separate')
    args=p.parse_args();assert not args.out.exists();args.out.mkdir(parents=True)
    torch.manual_seed(20261001);E=ops._get_esimd();E.exl3_set_int8(1)
    idx=json.loads((args.model/'model.safetensors.index.json').read_text())['weight_map']
    handles={}
    def get(key):
        file=idx[key]
        if file not in handles:handles[file]=safe_open(str(args.model/file),'pt',device='cpu')
        return handles[file].get_tensor(key)
    prefix='model.language_model.layers.'
    groups=[('gdn-qkvz',[prefix+'0.linear_attn.in_proj_qkv',prefix+'0.linear_attn.in_proj_z'],None),
            ('attention-qkv',[prefix+'3.self_attn.'+x+'_proj' for x in ('q','k','v')],None),
            ('gate-up',[prefix+'0.mlp.gate_proj',prefix+'0.mlp.up_proj'],None),
            ('down',[prefix+'0.mlp.down_proj'],None),
            ('head-first',['lm_head'],0),('head-last',['lm_head'],-512),('head-full',['lm_head'],None)]
    if args.groups:
        assert set(args.groups)<=set(v[0] for v in groups)
        groups=[g for g in groups if g[0] in args.groups]
    result=dict(status='RUNNING',torch=torch.__version__,seed=20261001,rows=ROWS,
                tolerance=TOLERANCE,groups=[],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                native_library_sha256=hashlib.sha256(Path(os.environ['EXL3_LIB']).read_bytes()).hexdigest(),
                checkpoint_index_sha256=hashlib.sha256((args.model/'model.safetensors.index.json').read_bytes()).hexdigest())
    def save():(args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        for name,keys,offset in groups:
            trs=[get(k+'.trellis') for k in keys]
            suhs=[get(k+'.suh') for k in keys];svhs=[get(k+'.svh') for k in keys]
            bits=trs[0].shape[-1]//16
            if offset is not None:
                n=svhs[0].numel();start=offset if offset>=0 else n+offset
                trs=[trs[0][:,start//16:(start+512)//16].contiguous()]
                svhs=[svhs[0][start:start+512]]
            widths=[t.shape[1]*16 for t in trs];bounds=[0]
            for n in widths:bounds.append(bounds[-1]+n)
            tr=torch.cat(trs,1).to('xpu');suh=torch.stack(suhs).to('xpu');svh=torch.cat(svhs).to('xpu')
            shard=torch.cat([torch.full((n//128,),i,dtype=torch.int32) for i,n in enumerate(widths)]).to('xpu')
            kdim=suh.shape[1];n=svh.numel();x=torch.randn((512,kdim),dtype=torch.float16,device='xpu')
            def linear(z):return E.linear(z,tr,suh,svh,shard,bounds,bits,2,128,16384)
            # Prime large INT8/oneDNN before every small/native shape.
            large=linear(x);torch.xpu.synchronize()
            fp32_gold=torch.empty((512,n),dtype=torch.float32,device='xpu')
            int8_pre=torch.empty((512,n),dtype=torch.float16,device='xpu')
            xh=torch.empty((len(keys),512,kdim),dtype=torch.float16,device='xpu')
            E.exl3_had_in_rm(x,suh,xh)
            for i,(left,right) in enumerate(zip(bounds,bounds[1:])):
                inner=ref.had_rows(x.float()*suh[i].float())
                q8=3.453125/127
                sx=xh[i].float().abs().amax(1,keepdim=True)/127
                xq=torch.round(xh[i].float()/sx).to(torch.int8)
                # Bound reference storage even for the full248320-row head.
                # Slice boundaries are128-aligned; output Hadamard blocks do
                # not cross them. This preserves every real head column.
                for start in range(left,right,16384):
                    stop=min(start+16384,right)
                    w=torch.empty((kdim,stop-start),dtype=torch.float16,device='xpu')
                    E.exl3_reconstruct(tr,w,start,bits,2)
                    fp32_gold[:,start:stop]=ref.had_rows(inner@w.float())*svh[start:stop].float()
                    wq=torch.round(w.float()/q8).to(torch.int8)
                    int8_pre[:,start:stop]=(torch._int_mm(xq,wq).float()*sx*q8).half()
                    del w,wq
                del inner,xq,sx
            int8_gold=torch.empty_like(int8_pre);E.exl3_had_out_h(int8_pre,svh,int8_gold)
            row=dict(name=name,keys=keys,bits=bits,k=kdim,n=n,head_slice_start=offset,cases=[],padding=[],compiled=[])
            result['groups'].append(row);save()
            for count in ROWS:
                eager=linear(x[:count]);gold=fp32_gold[:count] if count<=128 else int8_gold[:count]
                case=dict(rows=count,expected_family='SmallM' if count<=128 else 'INT8-prefill',reference=check(eager,gold))
                if count<=128:
                    # The frozen FULL_DECODE_ONLY profile captures at most64
                    # target rows. Cover128 additionally; INT8-prefill is eager.
                    for _ in range(2):linear(x[:count])
                    torch.xpu.synchronize();graph=torch.xpu.XPUGraph()
                    gx=x[:count].clone()
                    with torch.xpu.graph(graph):gy=linear(gx)
                    graph.replay();case['graph']=check(gy,gold)
                    gx.mul_(.75);graph.replay();case['mutated_graph']=check(gy,linear(gx))
                    del graph,gy,gx
                else:
                    case['graph_applicability']='Unsupported oneDNN INT8-prefill capture; frozen profile is FULL_DECODE_ONLY with maximum64 target rows. Eager numerical/compiled/padding gates remain required.'
                row['cases'].append(case);save()
            for count in ROWS:
                padded=128 if count<=128 else 512
                z=x[:padded].clone();base=linear(z)[:count].clone()
                z[count:]=torch.randn_like(z[count:])*64
                row['padding'].append(dict(actual=count,padded=padded,tail_poison=check(linear(z)[:count],base),
                                           same_family_active=check(base,linear(x[:count]))))
            # Isolate shape-class experiments: the same helper code object
            # otherwise accumulates closure specializations across all groups
            # and exhausts Dynamo's8-frame limit, unrelated to native dispatch.
            torch._dynamo.reset()
            compiled=torch.compile(linear,dynamic=True,fullgraph=True)
            compiled(x);torch.xpu.synchronize()
            for count in [1,4,20,64,128,129,512]:
                row['compiled'].append(dict(rows=count,after_large_first=check(compiled(x[:count]),linear(x[:count]))))
            if name=='attention-qkv':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.XPU]) as trace:
                    for count in [512,1,20,128,129]:compiled(x[:count])
                    torch.xpu.synchronize()
                trace.export_chrome_trace(str(args.out/'large-first-dispatch.trace.json'))
                row['trace_sha256']=hashlib.sha256((args.out/'large-first-dispatch.trace.json').read_bytes()).hexdigest()
                row['events']=[dict(name=v.key,count=v.count,device_time_total=v.device_time_total) for v in trace.key_averages()]
            print('PASS',name,'bits',bits,'shape',kdim,n,'eager/graph/padding/compiled',flush=True);save()
            del tr,suh,svh,shard,x,xh,large,fp32_gold,int8_pre,int8_gold,compiled
            gc.collect();torch.xpu.empty_cache()
        result['status']='PASS_EXPANDED_ROW_PADDING_GRAPH_GATE'
    except BaseException as exc:result['status']='FAILED';result['error']=repr(exc);raise
    finally:save()


if __name__=='__main__':main()

"""Bounded independent FP32 references for the FP16 SDPA boundary values."""
import argparse,json
from pathlib import Path
import torch
from exl3xpu.ops import _get_esimd

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists()
    torch.manual_seed(204410);op=_get_esimd();rows=[];guards=[];empty=torch.empty(0,device='xpu')
    for Q,L in [(64,4096),(256,4097),(256,8193)]:
        for family in ['zero','low_variance','large_constant','large_random']:
            q=torch.randn(6,Q,256,device='xpu',dtype=torch.float16)
            k=torch.randn(1,L,256,device='xpu',dtype=torch.float16);v=torch.randn_like(k)
            if family=='zero':q.zero_();k.zero_()
            elif family=='low_variance':q.mul_(.001).add_(1);k.mul_(.001).add_(1)
            elif family=='large_constant':q.fill_(32);k.fill_(32)
            else:q.mul_(4);k.mul_(4)
            out=torch.empty_like(q);before=[x.clone() for x in [q,k,v]]
            op.exl3_sdpa(q,k,v,out,empty,True,.0625);torch.xpu.synchronize()
            scores=q.float()@k[0].float().T*.0625
            visible=L-Q+torch.arange(Q,device='xpu')[:,None]
            scores.masked_fill_(torch.arange(L,device='xpu')[None]>visible,-float('inf'))
            gold=scores.softmax(-1)@v[0].float()
            record=dict(family=family,Q=Q,L=L,allclose=bool(torch.allclose(out.float(),gold,rtol=.01,atol=.003)),finite=bool(out.isfinite().all()),max_abs=float((out.float()-gold).abs().max()),inputs_unchanged=all(torch.equal(x,y) for x,y in zip([q,k,v],before)))
            rows.append(record);assert record['allclose'] and record['finite'] and record['inputs_unchanged'],record
    # These established input checks run before partition creation or pointer
    # use. Do not deliberately hand an XPU kernel an unsupported CPU output.
    bad=[('fp32_query',q.float(),k,v,.0625),('fp32_key',q,k.float(),v,.0625),('fp32_value',q,k,v.float(),.0625),('noncontiguous_query',q.transpose(1,2),k,v,.0625),('cpu_query',q.cpu(),k,v,.0625),('zero_scale',q,k,v,0),('negative_scale',q,k,v,-1),('nan_scale',q,k,v,float('nan'))]
    for name,x,y,z,scale in bad:
        try:op.exl3_sdpa(x,y,z,out,empty,True,scale)
        except (RuntimeError,ValueError) as error:guards.append(dict(name=name,rejected=True,error=str(error).splitlines()[0]))
        else:raise AssertionError('Invalid input accepted: '+name)
    torch.xpu.synchronize()
    report=dict(status='PASS_FP16_SDPA_EDGE_VALUES_AND_EXISTING_INPUT_GUARDS',rtol=.01,atol=.003,cases=rows,input_guards=guards,
        scope='Same FP16 query/key/value data, independent FP32 matmul/causal softmax/value reference. This is diagnostics, never throughput. Large-constant raw products exceed FP16 accumulation range; SDPA score descriptor remains FP32. BF16 execution is not used. Guards unchanged; unsupported CPU output pointers are never passed to a device kernel.')
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(report['status'],len(rows),len(guards),flush=True)
if __name__=='__main__':main()

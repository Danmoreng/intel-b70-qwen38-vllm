#!/usr/bin/env python3
"""Independent bitstream+FP32 oracle, all dispatch rows, graphs and poisoned tails.

The independent NumPy oracle explicitly unpacks MSB-first bits and circular
16-bit windows. It does not call the candidate decoder or reconstruction op.
Real checkpoint column slices cover each fusion boundary and both head ends.
"""
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from exl3xpu import ops, ref
from matrix_panel import load_layers, configure

ROWS=[1,2,3,4,5,7,8,9,15,16,17,20,24,25,31,32,33,40,41,48,49,64,65,128]
TOLERANCE=2e-3


def independent_inner(tr, bits):
    data=tr.cpu().contiguous().numpy()
    tk,tn,_=data.shape; flat=data.reshape(-1,16*bits)
    out=np.empty((flat.shape[0],256),dtype=np.float16)
    t=np.arange(256); nbits=256*bits
    windows=((t[:,None]+1)*bits-16+np.arange(16)[None,:])%nbits
    # Independent IEEE-half mul1 LUT: compute exact affine in float64, round once.
    state=np.arange(65536,dtype=np.uint64); mixed=state*np.uint64(0x83DCD12D)&np.uint64(0xffffffff)
    byte_sum=sum((mixed>>shift)&255 for shift in (0,8,16,24))
    lut=((1024+byte_sum).astype(np.float64)*np.float64(0.00676727294921875)-np.float64(10.3828125)).astype(np.float16)
    perm=np.asarray(ref.tensor_core_perm())
    for start in range(0,len(flat),256):
        half=flat[start:start+256].astype(np.uint16)
        words=half[:,0::2].astype(np.uint32)|(half[:,1::2].astype(np.uint32)<<16)
        unpacked=np.unpackbits(words.astype('>u4').view(np.uint8),axis=-1,bitorder='big')
        states=np.zeros((len(words),256),dtype=np.uint16)
        for bit in range(16):states=(states<<1)|unpacked[:,windows[:,bit]].astype(np.uint16)
        out[start:start+len(words),perm]=lut[states]
    return torch.from_numpy(out.reshape(tk,tn,16,16).transpose(0,2,1,3).reshape(tk*16,tn*16).copy())


def check(a,b):
    a=a.float();b=b.float();rel=float((a-b).norm()/b.norm().clamp_min(1e-20))
    assert bool(a.isfinite().all()) and rel<TOLERANCE,(rel,float((a-b).abs().max()))
    return dict(relative_norm=rel,max_abs=float((a-b).abs().max()))


def main():
    torch.set_num_threads(4);torch.manual_seed(33261109)
    E=ops._get_esimd();configure(E)
    layers=load_layers(Path('/models/checkpoint'))
    library=Path(os.environ.get('EXL3_LIB',Path(ops.__file__).with_name('_C.so')))
    out=Path('/results/result.json')
    result=dict(status='RUNNING',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        panel_source_sha256=hashlib.sha256(Path(__file__).with_name('matrix_panel.py').read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),torch=torch.__version__,
        fixed_tolerance=TOLERANCE,rows=ROWS,cases=[],scope='Actual shape-class representatives, real fusion boundaries and head ends. Independent CPU bitstream decoder+FP32 Hadamard/matmul; no universal full-model byte-identity claim.')
    def save():out.write_text(json.dumps(result,indent=2)+'\n')
    save();seen=set()
    try:
        for l in layers:
            cls=(l['name'],l['k'],l['n'],l['bits'])
            if cls in seen:continue
            seen.add(cls)
            x=torch.randn(128,l['k'],device='xpu',dtype=torch.float16)
            def linear(z):return E.linear(z,l['tr'],l['suh'],l['svh'],l['shard'],l['bounds'],l['bits'],2,128,16384)
            slices=set([0,l['n']-512])
            for left,right in zip(l['bounds'],l['bounds'][1:]):
                slices.add(left);slices.add(max(left,right-512))
            slices=sorted(slices)
            gold={}
            for start in slices:
                stop=min(start+512,l['n'])
                group=next(i for i,(a,b) in enumerate(zip(l['bounds'],l['bounds'][1:])) if a<=start<b)
                assert stop<=l['bounds'][group+1]
                tr=l['tr'][:,start//16:stop//16].cpu().contiguous()
                w=independent_inner(tr,l['bits']).to('xpu')
                # Exact decoder equality independently of matrix accumulation.
                native=torch.empty_like(w);E.exl3_reconstruct(l['tr'],native,start,l['bits'],2)
                assert torch.equal(native,w),('decode',l['name'],start)
                inner=ref.had_rows(x.float()*l['suh'][group].float())
                gold[start]=ref.had_rows(inner@w.float())*l['svh'][start:stop].float()
                del w,native,inner
            row=dict(name=l['name'],keys=l['keys'],k=l['k'],n=l['n'],bits=l['bits'],slices=slices,cases=[])
            result['cases'].append(row);save()
            for m in ROWS:
                eager=linear(x[:m]);torch.xpu.synchronize()
                checks={str(start):check(eager[:,start:start+g.shape[1]],g[:m]) for start,g in gold.items()}
                for _ in range(2):linear(x[:m])
                torch.xpu.synchronize();graph=torch.xpu.XPUGraph();gx=x[:m].clone()
                with torch.xpu.graph(graph):gy=linear(gx)
                graph.replay();torch.xpu.synchronize()
                graphchecks={str(start):check(gy[:,start:start+g.shape[1]],g[:m]) for start,g in gold.items()}
                gx.mul_(.75);graph.replay();torch.xpu.synchronize()
                mutation=check(gy,linear(gx))
                # An inactive row must not feed back into active rows at any row block.
                padded=x.clone(); base=linear(padded)[:m].clone();padded[m:]=torch.randn_like(padded[m:])*64
                padding=check(linear(padded)[:m],base)
                row['cases'].append(dict(rows=m,eager=checks,graph=graphchecks,mutated_graph=mutation,poisoned_padding=padding))
                del graph,gx,gy,eager,padded,base;gc.collect()
                save()
            print('PASS',l['name'],l['k'],l['n'],'rows',len(ROWS),'slices',len(slices),flush=True)
            del gold,x;gc.collect();torch.xpu.empty_cache()
        result['status']='PASS_INDEPENDENT_MATRIX_NUMERICS_GRAPHS_PADDING'
    except BaseException as e:result.update(status='FAILED',error=repr(e));raise
    finally:save()


if __name__=='__main__':main()

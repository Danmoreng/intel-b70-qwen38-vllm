#!/usr/bin/env python3
"""All actual target linears, vLLM fusion groups, native FULL graph replays.

This measures only matrix operators on seeded synthetic activations. It is not
an agent speed claim. Keep fused group order explicit, including the full head.
"""
import argparse
import collections
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

import torch
from safetensors import safe_open
from exl3xpu import ops


def load_layers(model):
    qc = json.loads((model/'quantization_config.json').read_text())['tensor_storage']
    idx = json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    handles = {}
    def get(name):
        f=idx[name]
        if f not in handles: handles[f]=safe_open(str(model/f),'pt',device='cpu')
        return handles[f].get_tensor(name)
    fuse={'in_proj_qkv':'in_proj_qkvz','in_proj_z':'in_proj_qkvz',
          'gate_proj':'gate_up','up_proj':'gate_up',
          'q_proj':'qkv','k_proj':'qkv','v_proj':'qkv'}
    order={'in_proj_qkvz':['in_proj_qkv','in_proj_z'], 'gate_up':['gate_proj','up_proj'],
           'qkv':['q_proj','k_proj','v_proj']}
    groups=collections.OrderedDict()
    for key,val in qc.items():
        if val.get('quant_format')!='exl3': continue
        base,_,leaf=key.rpartition('.')
        group=(base,fuse.get(leaf,leaf)); groups.setdefault(group,[]).append(key)
    layers=[]
    for (base,name),keys in groups.items():
        if name in order:
            assert {k.rsplit('.',1)[-1] for k in keys}==set(order[name])
            keys=sorted(keys,key=lambda k:order[name].index(k.rsplit('.',1)[-1]))
        trs=[get(k+'.trellis') for k in keys]
        widths=[t.shape[1]*16 for t in trs]; bounds=[0]
        for n in widths: bounds.append(bounds[-1]+n)
        layers.append(dict(name=name,keys=keys,
            tr=torch.cat(trs,1).to('xpu'),suh=torch.stack([get(k+'.suh') for k in keys]).to('xpu'),
            svh=torch.cat([get(k+'.svh') for k in keys]).to('xpu'),
            shard=torch.cat([torch.full((n//128,),i,dtype=torch.int32) for i,n in enumerate(widths)]).to('xpu'),
            bounds=bounds,bits=trs[0].shape[-1]//16,k=trs[0].shape[0]*16,n=sum(widths)))
    return layers


def configure(E, threads=1024, mb16=1408, mb64=2048, local=8):
    E.exl3_set_target_threads(threads); E.exl3_set_target_threads_mb16(mb16)
    E.exl3_set_target_threads_mb64(mb64); E.exl3_set_local(local)
    E.exl3_set_vec_max_m(2); E.exl3_set_max_mb(64); E.exl3_set_fused(0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=Path('/models/checkpoint'))
    p.add_argument('--out',type=Path,default=Path('/results'))
    p.add_argument('--tune',action='store_true')
    p.add_argument('--rows',type=int,nargs='+',default=[1,4,8,16,32,64])
    p.add_argument('--replays',type=int,default=12)
    p.add_argument('--fingerprints',action='store_true',help='Hash all actual input/output bytes after timing')
    a=p.parse_args(); torch.manual_seed(20261109); torch.set_num_threads(4)
    E=ops._get_esimd(); layers=load_layers(a.model)
    assert len(layers)>200
    library=Path(os.environ.get('EXL3_LIB',Path(ops.__file__).with_name('_C.so')))
    shapes=collections.Counter((l['name'],l['k'],l['n'],l['bits']) for l in layers)
    result=dict(status='RUNNING',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),torch=torch.__version__,
        device=torch.xpu.get_device_name(),fused_linears=len(layers),
        trellis_bytes=sum(l['tr'].numel()*2 for l in layers),
        shapes=[dict(name=k[0],k=k[1],n=k[2],bits=k[3],count=n) for k,n in shapes.items()],
        checkpoint_index_sha256=hashlib.sha256((a.model/'model.safetensors.index.json').read_bytes()).hexdigest(),
        scope='Only actual target EXL3 linears, synthetic activations, all401 source matrices fused as vLLM, full head. No attention, GDN, drafter, sampling or actual-agent benefit claim.',cases=[])
    a.out.mkdir(parents=True,exist_ok=True)
    def save(): (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        cases=[]
        for m in a.rows:
            configs=[{}]
            if a.tune and m in (4,8,16): configs += [dict(mb16=t) for t in (768,1024,1280,1536,1664)]
            if a.tune and m==32: configs += [dict(threads=t) for t in (512,768,1280,1536,2048)]
            if a.tune and m in (16,32): configs += [dict(local=v) for v in (4,16)]
            for config in configs: cases.append((m,config))
        random.Random(90611).shuffle(cases)
        for m,config in cases:
            configure(E,**config)
            xs={l['k']:torch.randn(m,l['k'],device='xpu',dtype=torch.float16) for l in layers}
            outs=[torch.empty((m,l['n']),device='xpu',dtype=torch.float16) for l in layers]
            def run():
                for l,o in zip(layers,outs):
                    E.exl3_gemm_small(xs[l['k']],l['tr'],l['suh'],l['svh'],l['shard'],o,l['bits'],2)
            for _ in range(2): run()
            torch.xpu.synchronize(); g=torch.xpu.XPUGraph()
            with torch.xpu.graph(g): run()
            for _ in range(3): g.replay()
            torch.xpu.synchronize()
            samples=[]
            for repeat in range(5):
                start=torch.xpu.Event(enable_timing=True); end=torch.xpu.Event(enable_timing=True)
                t=time.perf_counter();start.record()
                for _ in range(a.replays):g.replay()
                end.record();end.synchronize()
                samples.append(dict(gpu_ms=start.elapsed_time(end)/a.replays,
                                    wall_ms=(time.perf_counter()-t)*1000/a.replays))
            row=dict(rows=m,config=config,samples=samples,
                gpu_ms=statistics.median(v['gpu_ms'] for v in samples),
                wall_ms=statistics.median(v['wall_ms'] for v in samples),
                allocation_bytes=torch.xpu.memory_allocated(),reserved_bytes=torch.xpu.memory_reserved())
            assert all(torch.isfinite(o).all().item() for o in outs)
            if a.fingerprints:
                row['input_sha256']={str(k):hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for k,x in xs.items()}
                row['output_sha256']={str(i)+':'+l['keys'][0]:hashlib.sha256(o.cpu().numpy().tobytes()).hexdigest()
                                      for i,(l,o) in enumerate(zip(layers,outs))}
            result['cases'].append(row);save()
            print(json.dumps({k:v for k,v in row.items() if k not in ('input_sha256','output_sha256')}),flush=True)
            del g,outs,xs;gc.collect();torch.xpu.empty_cache()
        result['status']='PASS_ACTUAL_TARGET_MATRIX_PANEL_NOT_AGENT_BENEFIT'
    except BaseException as e:
        result.update(status='FAILED',error=repr(e));raise
    finally: save()


if __name__=='__main__':main()

"""Small mixed-precision operator gate using unchanged checkpoint weights."""
import json, random, statistics, time
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from vllm_xpu_kernels import _xpu_C

ROOT=Path('/work')
torch.ops.load_library('/src/runs/build/gdn_projection.so')
torch.manual_seed(714)
model=Path('/model')
index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
layers=sorted({int(k.split('.layers.')[1].split('.')[0]) for k in index if '.linear_attn.in_proj_b.weight' in k})
chosen=[layers[0],layers[len(layers)//2],layers[-1]]
def read(name):
    with safe_open(str(model/index[name]),framework='pt',device='cpu') as f:return f.get_tensor(name)
def weights(layer):
    prefix=f'model.language_model.layers.{layer}.linear_attn.'
    w=torch.cat([read(prefix+f'in_proj_{part}.qweight') for part in ['qkv','z']],dim=1).t().contiguous()
    s=torch.cat([read(prefix+f'in_proj_{part}.scales') for part in ['qkv','z']],dim=1).contiguous()
    ba=torch.cat([read(prefix+f'in_proj_{part}.weight') for part in ['b','a']],dim=0).contiguous()
    for part in ['qkv','z']:
        assert torch.equal(read(prefix+f'in_proj_{part}.g_idx'),torch.arange(5120,dtype=torch.int32)//128)
    return [x.to('xpu') for x in [w,s,ba]]
def timing(fn):
    torch.xpu.synchronize();a=torch.xpu.Event(enable_timing=True);b=torch.xpu.Event(enable_timing=True)
    start=time.perf_counter();a.record();fn();b.record();b.synchronize()
    return {'device_ms':a.elapsed_time(b),'wall_ms':(time.perf_counter()-start)*1000}
checks=[];rows=[]
zero=torch.tensor([8],device='xpu',dtype=torch.int8)
for layer in chosen:
    w,s,ba=weights(layer)
    for m in [1,2,3,4,5]:
        x=torch.randn(m,5120,device='xpu',dtype=torch.float16)
        control=lambda:(torch.ops._xpu_C.int4_gemm_w4a16(x,w.t(),None,s,zero,128,None),F.linear(x,ba))
        candidate=lambda:torch.ops.b70_gdn.project(x,w,s,ba)
        expected=control();actual=candidate();torch.xpu.synchronize()
        parts=[]
        for ref,got in zip(expected,actual):
            delta=(ref.float()-got.float()).abs()
            parts.append({'finite':bool(got.isfinite().all()),'allclose':torch.allclose(ref,got,atol=.002,rtol=.002),'max_abs':float(delta.max()),'relative_l2':float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(ref.float()).clamp_min(1e-9))})
        check={'layer':layer,'m':m,'parts':parts,'pass':all(p['finite'] and p['allclose'] and p['relative_l2']<.001 for p in parts)}
        checks.append(check);print('CHECK',json.dumps(check),flush=True)
        (ROOT/'correctness.json').write_text(json.dumps(checks,indent=2)+'\n')
        if not check['pass']:continue
        for fn in [control,candidate]:
            for _ in range(8):fn()
        functions={'control':control,'candidate':candidate}
        # Record eager and captured-graph timing separately; serving uses graphs.
        for mode in ['eager','graph']:
            graphs={}
            if mode=='graph':
                for label,fn in functions.items():
                    graph=torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph):result=fn()
                    graphs[label]=(graph,result)
                timed={label:value[0].replay for label,value in graphs.items()}
            else:timed=functions
            samples={label:[] for label in timed};rng=random.Random(layer*100+m)
            for _ in range(31):
                labels=list(timed);rng.shuffle(labels)
                for label in labels:samples[label].append(timing(timed[label]))
            medians={label:{k:statistics.median(v[k] for v in values) for k in ['device_ms','wall_ms']} for label,values in samples.items()}
            row={'layer':layer,'m':m,'mode':mode,'medians':medians,'speedup_percent':100*(medians['control']['device_ms']/medians['candidate']['device_ms']-1),'samples':samples}
            rows.append(row);(ROOT/'timings.json').write_text(json.dumps(rows,indent=2)+'\n')
            print('TIMING',json.dumps({k:v for k,v in row.items() if k!='samples'}),flush=True)
        del x,expected,actual
    del w,s,ba
gate=[r for r in rows if r['mode']=='graph' and r['m'] in [1,5]]
assessment={'correct':len(checks)==15 and all(c['pass'] for c in checks),'graph_m1_m5_speedups':[r['speedup_percent'] for r in gate],'scope':'real weights, seeded synthetic hidden states; not end-to-end serving'}
assessment['serving_worth_testing']=assessment['correct'] and len(gate)==6 and all(r['speedup_percent']>=3 for r in gate)
(ROOT/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
print('ASSESSMENT',json.dumps(assessment),flush=True)

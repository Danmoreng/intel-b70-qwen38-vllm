"""First operator screen: real norm weights, explicitly synthetic QKV inputs.

This screen cannot qualify production or substitute for real-activation replay.
Both arms use the installed model method and preserve split-view row strides.
"""
import json
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention, Qwen3NextRMSNorm
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

OUT = Path('/work')
MODEL = Path('/model')
index = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
config = json.loads((MODEL/'config.json').read_text())['text_config']
torch.manual_seed(1909)
torch.set_grad_enabled(False)

def weight(name):
    with safe_open(str(MODEL/index[name]), framework='pt', device='cpu') as f:
        return f.get_tensor(name).to(device='xpu', dtype=torch.float16)

def measure(fn, repeat=10):
    a, b = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    torch.xpu.synchronize()
    start = time.perf_counter()
    a.record()
    for _ in range(repeat): fn()
    b.record(); b.synchronize()
    return {'device_ms': a.elapsed_time(b)/repeat,
            'wall_ms': (time.perf_counter()-start)*1000/repeat}

def compare(expected, actual):
    rows=[]
    for label, ref, got in zip(['q','k','v','gate'], expected, actual):
        delta=(got.float()-ref.float()).abs()
        exact=torch.equal(ref,got)
        rows.append({'part':label, 'exact':exact,
            'finite':bool(got.isfinite().all()), 'max_abs':float(delta.max()),
            'relative_l2':float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(ref.float()).clamp_min(1e-12)),
            'pass':bool(got.isfinite().all()) and (exact if label in ('gate','v') else torch.allclose(ref,got,atol=.01,rtol=.002))})
    return rows

results=[]
with set_current_vllm_config(VllmConfig()):
    rope=MRotaryEmbedding(256,64,config['max_position_embeddings'],
                         config['rope_parameters']['rope_theta'],True,torch.float16,
                         mrope_section=[11,11,10],mrope_interleaved=True).to('xpu')
    for layer in ('3','35','63','mtp'):
        prefix = 'mtp.layers.0.self_attn.' if layer=='mtp' else f'model.language_model.layers.{layer}.self_attn.'
        qnorm=Qwen3NextRMSNorm(256,eps=config['rms_norm_eps']).to(device='xpu',dtype=torch.float16)
        knorm=Qwen3NextRMSNorm(256,eps=config['rms_norm_eps']).to(device='xpu',dtype=torch.float16)
        qnorm.weight.copy_(weight(prefix+'q_norm.weight'));knorm.weight.copy_(weight(prefix+'k_norm.weight'))
        shim=SimpleNamespace(q_size=6144,kv_size=1024,num_heads=24,num_kv_heads=4,
            head_dim=256,attn_output_gate=True,q_norm=qnorm,k_norm=knorm,rotary_emb=rope,
            _b70_xpu_qk_fusion=True)
        shim._b70_qk_metadata_ok=lambda q,p:Qwen3NextAttention._b70_qk_metadata_ok(shim,q,p)
        for m in (1,2,3,4,5,16,64,256,1663,1664,1665,3328,4096,4992,6656):
            qkv=torch.randn((m,14336),device='xpu',dtype=torch.float16)*3
            for position_mode in ('text','high','mrope'):
                pos=torch.arange(m,device='xpu',dtype=torch.int64)
                if position_mode=='high': pos=pos+200704-m
                if position_mode=='mrope':pos=torch.stack([pos,pos//11+37,pos%11+83])
                assert shim._b70_qk_metadata_ok(qkv,pos)
                def call(fused):
                    shim.use_fused_qk_norm_rope_gate=fused
                    return Qwen3NextAttention._project_qkv_gate(shim,qkv,pos)
                ref=call(False);got=call(True)
                parts=compare(ref,got)
                row={'layer':layer,'m':m,'positions':position_mode,'parts':parts,
                     'pass':all(x['pass'] for x in parts),'timing':[]}
                if row['pass'] and position_mode=='mrope':
                    functions={'control':lambda:call(False),'candidate':lambda:call(True)}
                    for fn in functions.values():
                        for _ in range(8):fn()
                    for mode in ('eager','graph'):
                        graphs={}
                        if mode=='graph':
                            for label,fn in functions.items():
                                graph=torch.xpu.XPUGraph()
                                with torch.xpu.graph(graph):output=fn()
                                graphs[label]=(graph,output)
                            timed={k:v[0].replay for k,v in graphs.items()}
                            for fn in timed.values():fn()
                            replay_parts=compare(graphs['control'][1],graphs['candidate'][1])
                            row['graph_pass']=all(x['pass'] for x in replay_parts)
                        else:timed=functions
                        samples={k:[] for k in timed}
                        rng=random.Random(1909+m)
                        for _ in range(21):
                            labels=list(timed);rng.shuffle(labels)
                            for label in labels:samples[label].append(measure(timed[label]))
                        medians={k:{metric:statistics.median(x[metric] for x in v) for metric in ('device_ms','wall_ms')} for k,v in samples.items()}
                        row['timing'].append({'mode':mode,'medians':medians,'samples':samples,
                            'latency_reduction_pct':100*(1-medians['candidate']['device_ms']/medians['control']['device_ms'])})
                results.append(row)
                (OUT/'operator-results.json').write_text(json.dumps(results,indent=2)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k!='timing'}),flush=True)
                # A numerical failure is sufficient to stop this implementation.
                if not row['pass'] or row.get('graph_pass') is False:
                    (OUT/'screen-decision.json').write_text(json.dumps({'pass':False,'reason':'numerical gate','case':row},indent=2))
                    raise SystemExit(2)
    graph_rows=[t for r in results for t in r['timing'] if t['mode']=='graph']
    decision={'pass':all(t['latency_reduction_pct']>2 for t in graph_rows),
              'cases':len(results),'min_graph_latency_reduction_pct':min(t['latency_reduction_pct'] for t in graph_rows),
              'scope':'Synthetic QKV, real norm weights; next gate requires captured activations and serving. No promotion.'}
    (OUT/'screen-decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps(decision),flush=True)

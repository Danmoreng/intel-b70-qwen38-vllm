"""Replay captured production projections, weights and position/cache tensors."""
from pathlib import Path
import json,statistics,time,random
from types import SimpleNamespace
import torch
from vllm.config import VllmConfig,set_current_vllm_config
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention,Qwen3NextRMSNorm
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

root=Path('/work');data=root/'capture/activations'
torch.set_grad_enabled(False)
results=[]
with set_current_vllm_config(VllmConfig()):
    for path in sorted(data.glob('*.pt')):
        if path.name=='cos-sin-cache.pt':continue
        d=torch.load(path,weights_only=True,map_location='cpu')
        r={**d['rope'],'dtype':torch.float16}
        rope=MRotaryEmbedding(**r).to('xpu')
        rope.cos_sin_cache=torch.load(data/'cos-sin-cache.pt',weights_only=True).to('xpu')
        qn=Qwen3NextRMSNorm(256,eps=d['eps']).to(device='xpu',dtype=torch.float16)
        kn=Qwen3NextRMSNorm(256,eps=d['eps']).to(device='xpu',dtype=torch.float16)
        qn.weight.copy_(d['qweight'].to('xpu'));kn.weight.copy_(d['kweight'].to('xpu'))
        qkv=d['qkv'].to('xpu');pos=d['positions'].to('xpu')
        shim=SimpleNamespace(q_size=6144,kv_size=1024,num_heads=24,num_kv_heads=4,head_dim=256,
            attn_output_gate=True,q_norm=qn,k_norm=kn,rotary_emb=rope,_b70_xpu_qk_fusion=True)
        shim._b70_qk_metadata_ok=lambda q,p:Qwen3NextAttention._b70_qk_metadata_ok(shim,q,p)
        assert shim._b70_qk_metadata_ok(qkv,pos)
        def call(flag):
            shim.use_fused_qk_norm_rope_gate=flag
            return Qwen3NextAttention._project_qkv_gate(shim,qkv,pos)
        ref=call(False);got=call(True)
        parts=[]
        for label,a,b in zip(('q','k','v','gate'),ref,got):
            diff=(a.float()-b.float()).abs()
            parts.append({'part':label,'max_abs':float(diff.max()),'relative_l2':float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(a.float()).clamp_min(1e-12)),
                'exact':torch.equal(a,b),'pass':bool(b.isfinite().all()) and (torch.equal(a,b) if label in ('v','gate') else torch.allclose(a,b,atol=.01,rtol=.002))})
        row={'file':path.name,'m':qkv.shape[0],'prefix':d['prefix'],'label':d['label'],
             'positions_distinct':pos.ndim==2 and not torch.equal(pos[0],pos[1]),'parts':parts,'pass':all(p['pass'] for p in parts)}
        graphs={}
        for label,flag in [('control',False),('candidate',True)]:
            for _ in range(8):call(flag)
            g=torch.xpu.XPUGraph()
            with torch.xpu.graph(g):outputs=call(flag)
            graphs[label]=(g,outputs)
        samples={k:[] for k in graphs}
        rng=random.Random(1910)
        for _ in range(21):
            order=list(graphs);rng.shuffle(order)
            for label in order:
                a,b=torch.xpu.Event(enable_timing=True),torch.xpu.Event(enable_timing=True)
                a.record()
                for _ in range(10):graphs[label][0].replay()
                b.record();b.synchronize();samples[label].append(a.elapsed_time(b)/10)
        row['graph_median_ms']={k:statistics.median(v) for k,v in samples.items()}
        row['graph_samples_ms']=samples
        row['graph_latency_reduction_pct']=100*(1-row['graph_median_ms']['candidate']/row['graph_median_ms']['control'])
        row['graph_pass']=all(torch.equal(a,b) if label in ('v','gate') else torch.allclose(a,b,atol=.01,rtol=.002) for label,a,b in zip(('q','k','v','gate'),graphs['control'][1],graphs['candidate'][1]))
        results.append(row)
        (root/'real-operator-results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='graph_samples_ms'}),flush=True)
        if not row['pass'] or not row['graph_pass']:break
    decision={'cases':len(results),'prefixes':sorted({r['prefix'] for r in results}),
        'real_mrope_distinct':any(r['positions_distinct'] for r in results),
        'pass':bool(results) and all(r['pass'] and r['graph_pass'] and r['graph_latency_reduction_pct']>2 for r in results)}
    decision['pass'] &= any('mtp' in r['prefix'] for r in results) and decision['real_mrope_distinct']
    (root/'real-decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps(decision),flush=True)

#!/usr/bin/env python3
"""Capture real EXL3 FP16 target auxiliaries; retain the 262144 context limit."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
LAYERS=[5,19,33,47,61]
class CaptureExtension:
    def install_dflash_capture(self):
        model=self.model_runner.model
        core=model.language_model.model if hasattr(model,'language_model') else model.model
        base=Path('/results/target-states');base.mkdir(exist_ok=True)
        tokens=torch.tensor([248070],device='xpu')
        torch.save(core.embed_input_ids(tokens).detach().cpu(),base/'mask-embedding.pt')
        counts={i:0 for i in LAYERS}
        def hook(idx):
            def save(module,args,kwargs,out):
                hidden,residual=out
                if hidden.shape[0]<64:return
                positions=kwargs.get('positions')
                assert positions is not None, (idx, len(args), list(kwargs))
                if positions.ndim == 2:
                    assert torch.equal(positions, positions[:1].expand_as(positions)), 'non-text mRoPE input'
                    positions = positions[0]
                assert positions.ndim == 1, positions.shape
                from vllm.forward_context import get_forward_context
                padding=get_forward_context().is_padding
                valid=(~padding) if padding is not None else torch.ones(hidden.shape[0],dtype=torch.bool,device=hidden.device)
                assert valid.shape[0]==hidden.shape[0]
                hidden=hidden[valid]
                residual=residual[valid] if residual is not None else None
                positions=positions[valid]
                combined=hidden if residual is None else hidden+residual
                assert combined.dtype in (torch.float16,torch.float32), str(combined.dtype)
                assert torch.isfinite(combined).all(), (idx,str(combined.dtype),float(combined.abs().max()))
                combined=combined.to(torch.float16)
                assert torch.isfinite(combined).all(), 'target auxiliary narrowing overflow'
                n=counts[idx];counts[idx]+=1
                torch.save(dict(hidden=combined.detach().cpu(),positions=positions.detach().cpu(),valid_rows=valid.detach().cpu()),base/f'layer-{idx}-chunk-{n}.pt')
            return save
        for i in LAYERS:core.layers[i].register_forward_hook(hook(i),with_kwargs=True)
        embedding_chunks=[0]
        def embeddings(module,args,out):
            ids=args[0]
            if out.shape[0]>=64:
                n=embedding_chunks[0];embedding_chunks[0]+=1
                torch.save(dict(ids=ids.detach().cpu(),embeddings=out.detach().cpu()),base/f'prompt-embeddings-{n}.pt')
        core.embed_tokens.register_forward_hook(embeddings)
        return dict(layers=LAYERS,scope='diagnostic eager hooks; target model arithmetic unchanged')

    def export_bonus_embedding(self, token_id):
        model=self.model_runner.model
        core=model.language_model.model if hasattr(model,'language_model') else model.model
        tokens=torch.tensor([token_id],device='xpu')
        torch.save(core.embed_input_ids(tokens).detach().cpu(),Path('/results/target-states/bonus-embedding.pt'))
        return token_id

def main():
    from vllm import LLM, SamplingParams
    p=argparse.ArgumentParser();p.add_argument('--device');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    engine=json.loads(Path('/engine-config.json').read_text())
    engine.update(dtype='float16',enforce_eager=True,compilation_config={'mode':0},async_scheduling=False)
    assert engine['max_model_len']==262144 and engine['speculative_config']['method']=='mtp'
    llm=LLM(model='/models/checkpoint',quantization='exl3',worker_extension_cls='capture_target.CaptureExtension',**engine)
    identity=llm.collective_rpc('install_dflash_capture')
    prompts=[
        'Explain the invariants of an LRU cache implementation in Python and provide tested code.\n' + 'cache hit miss eviction linked list hash map thread safety. '*400,
        'Write a clear explanation of RMS normalization, residual networks, floating point overflow, and lossless power-of-two scaling.\n' + 'Consider large and small activations and numerical cancellation. '*400,
    ]
    # Frozen single prefill chunks give paired context/embedding capture. Keep separate cases.
    receipts=[]
    for i,prompt in enumerate(prompts):
        base=Path('/results/target-states')
        result=llm.generate(prompt,SamplingParams(temperature=0,max_tokens=32),use_tqdm=False)[0]
        llm.collective_rpc('export_bonus_embedding',args=(result.outputs[0].token_ids[0],))
        files=list(base.glob('*.pt'))
        case=base/f'case-{i}';case.mkdir()
        for f in files:f.rename(case/f.name)
        # Save a new mask for the next request; do not re-register hooks.
        if i==0:torch.save(torch.load(case/'mask-embedding.pt',weights_only=True),base/'mask-embedding.pt')
        receipts.append(dict(case=i,prompt_tokens=len(result.prompt_token_ids),output_ids=result.outputs[0].token_ids,
             files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in case.glob('*.pt')}))
    a.out.write_text(json.dumps(dict(status='CAPTURED_REAL_TARGET_STATES_NOT_SERVING_BENCHMARK',engine=engine,identity=identity,cases=receipts),indent=2)+'\n')
if __name__=='__main__':main()

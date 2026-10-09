"""Actual W4 runtime role inventory, alias and numerical gates."""
import hashlib
import json
import os
from pathlib import Path

import torch
from exl3xpu.draft_w4 import audit_draft
from diagnostics import WorkerExtension as Diagnostics


class WorkerExtension(Diagnostics):
    def cache_ledger(self):
        from diagnostics import WorkerExtension as Diagnostics
        return Diagnostics.cache_ledger(self)

    def inspect_dflash_setup(self):
        runner = self.model_runner
        cfg = runner.vllm_config
        draft = runner.speculator.model
        target = runner.model.get_language_model() if hasattr(runner.model,'get_language_model') else runner.model
        core = getattr(target,'model',target)
        assert cfg.model_config.dtype == cfg.speculative_config.draft_model_config.dtype == torch.float16
        assert draft.model.embed_tokens is core.embed_tokens and draft.lm_head is target.lm_head
        assert not hasattr(draft.lm_head,'exl3_draft')
        assert draft.model.embed_tokens.weight.untyped_storage().data_ptr() == core.embed_tokens.weight.untyped_storage().data_ptr()
        eps = cfg.speculative_config.draft_model_config.hf_config.rms_norm_eps
        for layer in draft.model.layers:
            assert layer.residual_scale == 1/16
            assert layer.input_layernorm.variance_epsilon == layer.post_attention_layernorm.variance_epsilon == eps/256
        assert draft.model.norm.variance_epsilon == eps/256 and draft.model.hidden_norm.variance_epsilon == eps
        result = audit_draft(runner)
        from exl3xpu import shared_kv_verify as verify
        library = Path(os.environ['EXL3_M04_LIBRARY'])
        assert hashlib.sha256(library.read_bytes()).hexdigest() == os.environ['EXL3_M04_LIBRARY_SHA256']
        inventory = []
        storage_inventory=[];storages={}
        for kind,items in [('parameter',draft.named_parameters()),('buffer',draft.named_buffers())]:
            for name,tensor in items:
                if name.startswith(('lm_head.','model.embed_tokens.')):continue
                storage=tensor.untyped_storage();key=(str(tensor.device),storage.data_ptr())
                if key not in storages:storages[key]=dict(index=len(storages),bytes=storage.nbytes())
                storage_inventory.append(dict(kind=kind,name=name,shape=list(tensor.shape),dtype=str(tensor.dtype),
                    logical_bytes=tensor.numel()*tensor.element_size(),storage_index=storages[key]['index'],
                    storage_bytes=storage.nbytes(),storage_offset=tensor.storage_offset(),stride=list(tensor.stride())))
        for name, parameter in draft.named_parameters():
            if name.startswith(('lm_head.','model.embed_tokens.')):
                continue
            inventory.append(dict(name=name,shape=list(parameter.shape),dtype=str(parameter.dtype),
                                  bytes=parameter.numel()*parameter.element_size()))
        result.update(target_dtype='float16',draft_dtype='float16',draft_parameters=inventory,
            shared_target_embed=True,shared_full_target_head=True,max_model_len=cfg.model_config.max_model_len,
            num_speculative_tokens=cfg.speculative_config.num_speculative_tokens,
            draft_sample_method=cfg.speculative_config.draft_sample_method,
            m04_wrapper_sha256=hashlib.sha256(Path(verify.__file__).read_bytes()).hexdigest(),
            m04_library_sha256=os.environ['EXL3_M04_LIBRARY_SHA256'],
            draft_logical_parameter_bytes=sum(p['bytes'] for p in inventory),
            draft_registered_tensor_storage_inventory=storage_inventory,
            draft_registered_unique_storage_bytes=sum(v['bytes'] for v in storages.values()),
            storage_scope='Draft parameters and registered buffers, excluding shared target embed/head; aliases deduplicated. Direct speculator buffers and allocator/graph residency are separately reported by cache_ledger.',
            context_projection='packed KV-only' if os.environ.get('B70_DFLASH_KV_ONLY')=='1' else 'full quantized QKV, discard Q')
        Path('/results/draft-identity.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

    def w4_numerical_probe(self):
        torch.manual_seed(20261009)
        checks = []
        for name, module in self.model_runner.speculator.model.named_modules():
            if not hasattr(module,'draft_w4_loader_report'):
                continue
            k, n = module.qweight.shape[0]*8, module.qweight.shape[1]
            x = torch.randn(8, k, device='xpu', dtype=torch.float16)*.3
            actual = module.quant_method.apply(module, x).float().cpu()
            assert bool(torch.isfinite(actual).all())
            # Independent FP64 dot product for columns spanning every packed
            # constituent. Decode offset-binary nibbles without dense residency.
            offsets = [0]
            for size in module.draft_w4_sizes: offsets.append(offsets[-1]+size)
            columns = sorted({a for a in offsets[:-1]}|{b-1 for b in offsets[1:]}|
                             {(a+b)//2 for a,b in zip(offsets[:-1],offsets[1:])})
            packed = module.qweight[:, columns].cpu().long()
            shifts = torch.arange(0,32,4,dtype=torch.int64)
            integers = ((packed[:,None,:] >> shifts[None,:,None])&15).reshape(k,-1)-8
            scale = module.scales[:, columns].cpu().double().repeat_interleave(128,0)
            reference = x.cpu().double() @ (integers.double()*scale)
            error = (actual[:,columns].double()-reference).abs()
            # Frozen bound for FP16 dot/output rounding, independent of results.
            assert torch.allclose(actual[:,columns].double(), reference, rtol=.01, atol=.01), (name,float(error.max()))
            kv_check=None
            if hasattr(module,'draft_w4_kv_weight'):
                kv=module.quant_method.apply_context_kv(module,x).float().cpu()
                q_size=module.draft_w4_sizes[0]
                assert torch.allclose(kv,actual[:,q_size:],rtol=.005,atol=.005),name
                shared=module.draft_w4_kv_weight.untyped_storage().data_ptr()==module.qweight.untyped_storage().data_ptr()
                assert shared
                difference=(kv-actual[:,q_size:]).abs()
                kv_check=dict(status='PASS_KV_ONLY_VS_FULL_PACKED_QKV',
                    max_abs=float(difference.max()),rms=float(difference.square().mean().sqrt()),
                    rtol=.005,atol=.005,packed_storage_alias=shared,
                    weight_storage_offset=module.draft_w4_kv_weight.storage_offset())
            checks.append(dict(module=name,shape=[n,k],rows=8,columns=columns,
                max_abs=float(error.max()),rms=float(error.square().mean().sqrt()),rtol=.01,atol=.01,
                kv_only=kv_check))
        assert len(checks)==21
        kv_count=sum(c['kv_only'] is not None for c in checks)
        assert kv_count==(5 if os.environ.get('B70_DFLASH_KV_ONLY')=='1' else 0)
        result = dict(status='PASS_ACTUAL_DRAFT_W4_PACKED_KERNEL_COLUMNS',checks=checks,
                      actual_kv_only_projection_checks=kv_count,
                      scope='FP64 independent packed coefficient dot products across all runtime modules and fused shards; not draft acceptance qualification')
        Path('/results/w4-numerical-probe.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

    def w4_prefill_kv_probe(self):
        assert os.environ.get('B70_DFLASH_KV_ONLY')=='1'
        torch.manual_seed(20261010);checks=[]
        for name,module in self.model_runner.speculator.model.named_modules():
            if not hasattr(module,'draft_w4_kv_weight'):continue
            q_size=module.draft_w4_sizes[0]
            columns=[0,127,128,511,512,1023,1024,1151,1152,1535,1536,2047]
            for rows in [64,4096]:
                x=torch.randn(rows,5120,device='xpu',dtype=torch.float16)*.3
                full=module.quant_method.apply(module,x)
                assert full.dtype==torch.float16
                full_kv=full[:,q_size:].float().cpu();del full
                kv=module.quant_method.apply_context_kv(module,x)
                assert kv.dtype==torch.float16
                actual=kv.float().cpu();del kv
                assert torch.allclose(actual,full_kv,rtol=.005,atol=.005),(name,rows)
                selected=[0,1,7,8,31,32,rows-2,rows-1]
                packed=module.qweight[:,[q_size+c for c in columns]].cpu().long()
                shifts=torch.arange(0,32,4,dtype=torch.int64)
                integers=((packed[:,None,:]>>shifts[None,:,None])&15).reshape(5120,-1)-8
                scales=module.scales[:,[q_size+c for c in columns]].cpu().double().repeat_interleave(128,0)
                reference=x[selected].cpu().double()@(integers.double()*scales)
                observed=actual[selected][:,columns].double()
                assert torch.allclose(observed,reference,rtol=.01,atol=.01),(name,rows)
                checks.append(dict(module=name,rows=rows,selected_rows=selected,columns=columns,
                    kv_vs_full_max_abs=float((actual-full_kv).abs().max()),kv_rtol=.005,kv_atol=.005,
                    independent_dot_max_abs=float((observed-reference).abs().max()),dot_rtol=.01,dot_atol=.01,
                    input_dtype='float16',output_dtype='float16'))
        assert len(checks)==10
        result=dict(status='PASS_ACTUAL_PACKED_PREFILL_KV_ONLY_SHAPES',checks=checks,
            scope='64/4096 context rows, all five actual packed K/V projections; no production speed claim')
        Path('/results/w4-prefill-kv-probe.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

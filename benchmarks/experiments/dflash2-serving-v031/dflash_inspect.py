"""Actual worker attestation; invoked once before timing, never on decode."""
import hashlib
import json
from pathlib import Path
import torch
class WorkerExtension:
    def inspect_dflash_setup(self):
        runner=self.model_runner;draft=runner.speculator.model
        config=runner.vllm_config
        target=runner.model.get_language_model() if hasattr(runner.model,'get_language_model') else runner.model
        core=getattr(target,'model',target)
        assert config.model_config.dtype==torch.float16
        assert config.speculative_config.draft_model_config.dtype==torch.float16
        assert draft.model.quant_config is None, 'expected unquantized draft body'
        assert draft.model.embed_tokens is core.embed_tokens
        assert draft.lm_head is target.lm_head, 'must share the full EXL3 target head'
        assert not hasattr(draft.lm_head,'exl3_draft'), 'MTP vocabulary pruning must not affect DFlash'
        eps=config.speculative_config.draft_model_config.hf_config.rms_norm_eps
        assert len(draft.model.layers)==5
        for layer in draft.model.layers:
            assert layer.residual_scale==1/16
            assert layer.input_layernorm.variance_epsilon==eps/256
            assert layer.post_attention_layernorm.variance_epsilon==eps/256
        assert draft.model.norm.variance_epsilon==eps/256
        assert draft.model.hidden_norm.variance_epsilon==eps
        inventory=[];parameters=0
        for name,p in draft.named_parameters():
            if name.startswith(('model.embed_tokens.','lm_head.')):continue
            assert p.dtype==torch.float16,(name,str(p.dtype))
            flat=p.detach().flatten()
            for start in range(0,flat.numel(),1024*1024):
                assert torch.isfinite(flat[start:start+1024*1024]).all(),name
            parameters+=p.numel();inventory.append(dict(name=name,shape=list(p.shape),dtype=str(p.dtype)))
        assert parameters==1924404480+5120,parameters # dedicated unused mask placeholder
        result=dict(status='PASS_ACTUAL_FP16_DFLASH2_WORKER',target_dtype=str(config.model_config.dtype),draft_dtype=str(config.speculative_config.draft_model_config.dtype),draft_body_quantization=None,draft_parameters_including_mask_placeholder=parameters,shared_target_embed=True,shared_full_target_head=True,draft_class=type(draft).__name__,speculator_class=type(runner.speculator).__name__,max_model_len=config.model_config.max_model_len,num_speculative_tokens=config.speculative_config.num_speculative_tokens,parameters=inventory)
        assert result['draft_class']=='DFlash2Qwen3ForCausalLM' and result['speculator_class']=='DFlash2Speculator'
        Path('/results/draft-identity.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

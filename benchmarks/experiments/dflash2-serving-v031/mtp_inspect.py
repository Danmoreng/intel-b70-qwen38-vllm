"""Actual FP16 checkpoint-native MTP candidate attestation before timing."""
import hashlib
import json
import os
from pathlib import Path
import torch

class WorkerExtension:
    def inspect_mtp_setup(self):
        runner = self.model_runner
        config = runner.vllm_config
        draft = runner.speculator.model
        assert config.speculative_config.method == 'mtp'
        assert config.model_config.dtype == config.speculative_config.draft_model_config.dtype == torch.float16
        assert 'MTP' in type(draft).__name__.upper()
        assert config.quant_config.get_name() == 'exl3'
        library = Path(os.environ['EXL3_M04_LIBRARY'])
        digest = hashlib.sha256(library.read_bytes()).hexdigest()
        assert digest == os.environ['EXL3_M04_LIBRARY_SHA256']
        result = dict(status='PASS_ACTUAL_FP16_CHECKPOINT_NATIVE_MTP_WORKER',
                      target_dtype=str(config.model_config.dtype), draft_dtype=str(config.speculative_config.draft_model_config.dtype),
                      draft_class=type(draft).__name__, speculator_class=type(runner.speculator).__name__,
                      quantization=config.quant_config.get_name(), max_model_len=config.model_config.max_model_len,
                      num_speculative_tokens=config.speculative_config.num_speculative_tokens,
                      m04_library_sha256=digest, external_dflash_weights_loaded=False)
        Path('/results/draft-identity.json').write_text(json.dumps(result, indent=2) + '\n')
        return result

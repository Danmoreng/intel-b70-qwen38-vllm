"""Validate the installed vLLM configuration without loading model weights."""
import json
import sys
from vllm.engine.arg_utils import EngineArgs

dtype = sys.argv[1]
args = EngineArgs(
    model="mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16",
    revision="a47b0c6f0d756bc394c4cc629d5b0ded1acc7001",
    quantization="gptq", dtype=dtype, max_model_len=200704,
    gpu_memory_utilization=0.93, kv_cache_dtype="fp8", max_num_seqs=1,
    max_num_batched_tokens=4096, enable_prefix_caching=True,
    mamba_cache_mode="align", limit_mm_per_prompt={"image": 1, "video": 0},
    speculative_config={"method": "dflash", "model": "/draft",
                        "num_speculative_tokens": 7},
)
try:
    config = args.create_engine_config()
    print(json.dumps({"config_valid": True, "target_dtype": str(config.model_config.dtype),
        "draft_dtype": str(config.speculative_config.draft_model_config.dtype),
        "draft_architectures": config.speculative_config.draft_model_config.architectures,
        "v2_runner": config.use_v2_model_runner,
        "prefix_caching": config.cache_config.enable_prefix_caching,
        "context": config.model_config.max_model_len,
        "note": "Configuration only; no model load, correctness, capacity or speed claim."}))
except Exception as error:
    print(json.dumps({"config_valid": False, "dtype": dtype,
                      "error": str(error), "type": type(error).__name__}))
    raise

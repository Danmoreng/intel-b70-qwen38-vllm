"""Check the streamed forward against an ordinary Transformers forward.

Small randomized hybrid model contains both attention kinds. CPU-only test
needs no production interruption or large checkpoint.
"""
import json
from pathlib import Path
import tempfile

from safetensors.torch import save_file
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from stream_reference import stream_hidden


@torch.inference_mode()
def main():
    torch.manual_seed(20261001)
    torch.set_num_threads(2)
    cfg = Qwen3_5TextConfig(vocab_size=512, hidden_size=128, intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=32, linear_value_head_dim=32,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000000,
                         "partial_rotary_factor": .25, "mrope_section": [1, 1, 2],
                         "mrope_interleaved": True})
    cfg._attn_implementation = "sdpa"
    model = Qwen3_5ForCausalLM(cfg).eval()
    ids = torch.randint(0, 512, (2, 73))
    expected = model(ids, use_cache=False).logits
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        state = {}
        for name, tensor in model.state_dict().items():
            key = "model.language_model." + name[len("model."):] if name.startswith("model.") else name
            state[key] = tensor.contiguous()
        save_file(state, root / "weights.safetensors")
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "weights.safetensors" for k in state}}))
        (root / "config.json").write_text(json.dumps({"text_config": cfg.to_dict()}))
        hidden, weights, _ = stream_hidden(root, ids, torch.float32, "cpu", batch_size=1)
        actual = torch.nn.functional.linear(hidden, weights.tensor("lm_head.weight", "cpu", torch.float32))
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        print(json.dumps({"stream_matches_official_forward": True,
                          "max_logit_error": float((actual - expected).abs().max()),
                          "both_attention_types": True, "sequence_length": 73}))


if __name__ == "__main__":
    main()

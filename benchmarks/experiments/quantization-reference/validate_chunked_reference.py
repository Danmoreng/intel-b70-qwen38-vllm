"""Compare cached layer/token streaming with an ordinary hybrid-model forward."""
import json
from pathlib import Path
import tempfile

from safetensors.torch import save_file
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from stream_long_reference import stream_long_hidden


@torch.inference_mode()
def main():
    torch.manual_seed(20261001)
    torch.set_num_threads(2)
    cfg = Qwen3_5TextConfig(vocab_size=512, hidden_size=128, intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=32, linear_value_head_dim=32,
        layer_types=['linear_attention', 'full_attention', 'linear_attention', 'full_attention'],
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000000,
                         'partial_rotary_factor': .25, 'mrope_section': [1, 1, 2],
                         'mrope_interleaved': True})
    cfg._attn_implementation = 'sdpa'
    model = Qwen3_5ForCausalLM(cfg).eval()
    sequences = [torch.randint(0, 512, (1, length)) for length in (257, 385)]
    expected = [model(ids, use_cache=False).logits for ids in sequences]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        state = {}
        for name, tensor in model.state_dict().items():
            key = 'model.language_model.' + name[len('model.'):] if name.startswith('model.') else name
            state[key] = tensor.contiguous()
        save_file(state, root / 'weights.safetensors')
        (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {k: 'weights.safetensors' for k in state}}))
        (root / 'config.json').write_text(json.dumps({'text_config': cfg.to_dict()}))
        errors = {}
        for chunk in (64, 128):
            hidden, norm, weights, _ = stream_long_hidden(root,
                [ids[0].tolist() for ids in sequences], torch.float32, 'cpu', chunk_tokens=chunk)
            head = weights.tensor('lm_head.weight', 'cpu', torch.float32)
            for i, h in enumerate(hidden):
                actual = torch.nn.functional.linear(norm(h), head)
                torch.testing.assert_close(actual, expected[i], atol=3e-6, rtol=3e-5)
                errors[f'chunk{chunk}-length{len(sequences[i][0])}'] = float((actual - expected[i]).abs().max())
        print(json.dumps({'status': 'PASS', 'ordinary_forward_matches_cached_stream': True,
                          'both_attention_types': True, 'two_distinct_full_attention_indices': True,
                          'final_partial_chunk_and_single_token_state': True,
                          'tolerance_atol': 3e-6, 'tolerance_rtol': 3e-5,
                          'maximum_logit_errors': errors}), flush=True)


if __name__ == '__main__':
    main()

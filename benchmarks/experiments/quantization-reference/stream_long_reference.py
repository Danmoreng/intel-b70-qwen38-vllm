"""BF16 suffix reference with one resident original layer and bounded token chunks.

Use official Transformers layers, causal masks and DynamicCache for each layer.
Attention KV and GDN conv/recurrent state persist across chunks within that
layer and are released before loading the next layer. No truncated context.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding,
)
from stream_reference import Weights


def apply_chunked_layer(layer, hidden, cfg, rotary, index, chunk_tokens):
    if chunk_tokens < 64 or chunk_tokens % 64:
        raise ValueError('Chunk size must preserve GDN 64-token chunk boundaries')
    cache = DynamicCache(config=cfg)
    full_attention = cfg.layer_types[index] == 'full_attention'
    # Bound a potential FP32 SDPA math score matrix, too. Do not assume that
    # the installed XPU backend fuses arbitrary cached non-square masks.
    if full_attention:
        max_rows = (2 * 2**30) // (cfg.num_attention_heads * hidden.shape[1] * 4)
        chunk_tokens = min(chunk_tokens, max(64, (max_rows // 64) * 64))
    for start in range(0, hidden.shape[1], chunk_tokens):
        end = min(hidden.shape[1], start + chunk_tokens)
        h = hidden[:, start:end]
        pos = torch.arange(start, end, device=h.device).view(1, 1, -1).expand(3, 1, -1)
        pe = rotary(h, pos) if full_attention else (None, None)
        mask = create_causal_mask(config=cfg, inputs_embeds=h, attention_mask=None,
                                  past_key_values=cache, position_ids=pos[0], layer_idx=index) if full_attention else None
        hidden[:, start:end] = layer(h, position_embeddings=pe, attention_mask=mask,
                                    position_ids=pos[0], past_key_values=cache, use_cache=True)
    del cache
    return hidden


@torch.inference_mode()
def stream_long_hidden(root, all_ids, dtype, device, chunk_tokens=4096, progress=None):
    raw = json.loads((Path(root) / 'config.json').read_text())
    cfg = Qwen3_5TextConfig(**raw.get('text_config', raw))
    cfg._attn_implementation = 'sdpa'
    weights = Weights(root)
    embedding = weights.tensor('model.language_model.embed_tokens.weight', device, dtype)
    hidden = [F.embedding(torch.tensor(ids, device=device).view(1, -1), embedding) for ids in all_ids]
    del embedding
    gc.collect()
    if device.startswith('xpu'):
        torch.xpu.empty_cache()
    rotary = Qwen3_5TextRotaryEmbedding(cfg, device=device)
    for index in range(cfg.num_hidden_layers):
        start = time.monotonic()
        with torch.device('meta'):
            layer = Qwen3_5DecoderLayer(cfg, index)
        weights.module(layer, f'model.language_model.layers.{index}.', device, dtype)
        for window, h in enumerate(hidden):
            apply_chunked_layer(layer, h, cfg, rotary, index, chunk_tokens)
            if progress:
                if device.startswith('xpu'):
                    torch.xpu.synchronize()
                progress({'layer': index, 'window': window, 'tokens': h.shape[1],
                          'elapsed_layer_s': time.monotonic() - start,
                          'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                          'peak_xpu_gib': torch.xpu.max_memory_allocated() / 2**30 if device.startswith('xpu') else None})
        del layer
        gc.collect()
        if device.startswith('xpu'):
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
    with torch.device('meta'):
        norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    weights.module(norm, 'model.language_model.norm.', device, dtype)
    # Normalize only suffix prediction states. Retain full prefix through all layers.
    return hidden, norm, weights, cfg


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--panel', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chunk-tokens', type=int, default=4096)
    parser.add_argument('--dtype', choices=('bfloat16', 'float16'), default='bfloat16')
    parser.add_argument('--device', default='xpu')
    parser.add_argument('--windows', type=int, default=0)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError('Fresh reference output required')
    args.out.mkdir(parents=True, exist_ok=True)
    panel_bytes = Path(args.panel).read_bytes()
    panel = json.loads(panel_bytes)
    windows = panel['windows'][:args.windows] if args.windows else panel['windows']
    started = time.monotonic()
    hidden, norm, weights, cfg = stream_long_hidden(args.model, [w['ids'] for w in windows],
        getattr(torch, args.dtype), args.device, args.chunk_tokens,
        progress=lambda row: print(json.dumps(row), flush=True))
    head = weights.tensor('lm_head.weight', args.device, getattr(torch, args.dtype))
    summaries = []
    for index, window in enumerate(windows):
        first = window['prefix_tokens'] - 1
        last = len(window['ids']) - 1
        targets = torch.tensor(window['ids'][first + 1:last + 1], device=args.device)
        suffix_hidden = norm(hidden[index][:, first:last])[0]
        nll, selected = [], {}
        for start in range(0, len(targets), 64):
            end = min(start + 64, len(targets))
            logprobs = F.log_softmax(F.linear(suffix_hidden[start:end], head).float(), -1)
            assert torch.isfinite(logprobs).all()
            nll.extend((-logprobs.gather(1, targets[start:end, None]).squeeze(1)).cpu().tolist())
            for position in window['kl_positions']:
                if first + start <= position < first + end:
                    selected[position] = logprobs[position - first - start].cpu().numpy()
        assert len(nll) == window['suffix_tokens'] and set(selected) == set(window['kl_positions'])
        np.save(args.out / f'window-{index:03d}-nll.npy', np.asarray(nll, dtype=np.float32))
        np.save(args.out / f'window-{index:03d}-logprobs.npy', np.stack([selected[p] for p in window['kl_positions']]))
        summaries.append({'name': window['name'], 'domain': window['domain'],
                          'prefix_tokens': window['prefix_tokens'], 'suffix_tokens': len(nll),
                          'nll_mean': float(np.mean(nll)), 'perplexity': float(np.exp(np.mean(nll)))})
    result = {'kind': 'original_layer_and_token_chunk_stream', 'dtype': args.dtype,
              'panel_sha256': hashlib.sha256(panel_bytes).hexdigest(), 'windows': summaries,
              'vocab_size': cfg.vocab_size, 'chunk_tokens': args.chunk_tokens,
              'seconds': time.monotonic() - started, 'torch': torch.__version__,
              'model_path': args.model, 'context_truncated': False,
              'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
              'peak_xpu_gib': torch.xpu.max_memory_allocated() / 2**30 if args.device.startswith('xpu') else None}
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

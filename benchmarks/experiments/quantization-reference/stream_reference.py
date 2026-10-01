"""Unquantized Qwen3.5-architecture text reference, one resident layer at a time.

Uses the installed Transformers decoder, masks and rotary embeddings unchanged.
All windows traverse each loaded layer before loading the next. No KV cache,
speculation, image encoder, or disk copy of checkpoint weights is required.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from safetensors import safe_open
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding,
)
from transformers.masking_utils import create_causal_mask


class Weights:
    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(self, key, device, dtype):
        # Mapping lifetime is limited to this tensor; copying avoids accumulating
        # resident mmap pages for all 55 GB of weights in a 32 GB RAM machine.
        with safe_open(self.root / self.index[key], framework="pt", device="cpu") as f:
            src = f.get_tensor(key)
            result = src.to(device=device, dtype=dtype, copy=True)
        return result

    def module(self, module, prefix, device, dtype):
        state = {name: self.tensor(prefix + name, device, dtype)
                 for name in module.state_dict()}
        module.load_state_dict(state, strict=True, assign=True)
        return module.eval()


def stream_hidden(root, ids, dtype, device, batch_size=1, progress=None):
    raw = json.loads((Path(root) / "config.json").read_text())
    cfg = Qwen3_5TextConfig(**raw.get("text_config", raw))
    cfg._attn_implementation = "sdpa"
    weights = Weights(root)
    embedding = weights.tensor("model.language_model.embed_tokens.weight", device, dtype)
    hidden = F.embedding(ids.to(device), embedding)
    del embedding
    gc.collect()
    if device.startswith("xpu"):
        torch.xpu.empty_cache()
    pos = torch.arange(ids.shape[1], device=device).view(1, 1, -1).expand(3, batch_size, -1)
    text_pos = pos[0]
    rotary = Qwen3_5TextRotaryEmbedding(cfg, device=device)
    pe = rotary(hidden[:batch_size], pos)
    mask = create_causal_mask(config=cfg, inputs_embeds=hidden[:batch_size],
                              attention_mask=None, past_key_values=None,
                              position_ids=text_pos)
    for i in range(cfg.num_hidden_layers):
        started = time.monotonic()
        with torch.device("meta"):
            layer = Qwen3_5DecoderLayer(cfg, i)
        layer = weights.module(layer, f"model.language_model.layers.{i}.", device, dtype)
        for start in range(0, len(hidden), batch_size):
            h = hidden[start:start + batch_size]
            n = len(h)
            hidden[start:start + n] = layer(
                h, position_embeddings=tuple(t[:n] for t in pe),
                attention_mask=mask if cfg.layer_types[i] == "full_attention" else None,
                position_ids=text_pos[:n], past_key_values=None, use_cache=False)
        del layer
        gc.collect()
        if device.startswith("xpu"):
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
        if progress:
            progress({"layer": i, "seconds": round(time.monotonic() - started, 3),
                      "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
                      "peak_xpu_gib": round(torch.xpu.max_memory_allocated() / 2**30, 3)
                      if device.startswith("xpu") else None})
    with torch.device("meta"):
        norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    weights.module(norm, "model.language_model.norm.", device, dtype)
    return norm(hidden), weights, cfg


@torch.inference_mode()
def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    panel_bytes = Path(args.panel).read_bytes()
    panel = json.loads(panel_bytes)
    windows = panel["windows"][:args.windows] if args.windows else panel["windows"]
    ids = torch.tensor([w["ids"] for w in windows], dtype=torch.long)
    dtype = getattr(torch, args.dtype)
    started = time.monotonic()
    hidden, weights, cfg = stream_hidden(args.model, ids, dtype, args.device, args.batch_size,
        progress=lambda d: print(json.dumps(d), flush=True))
    head = weights.tensor("lm_head.weight", args.device, dtype)
    nlls, rows = [], []
    for i, w in enumerate(windows):
        full_nll = []
        selected = {}
        for start in range(0, len(w["ids"]) - 1, 64):
            end = min(start + 64, len(w["ids"]) - 1)
            logits = F.linear(hidden[i, start:end], head).float()
            lp = F.log_softmax(logits, dim=-1)
            assert torch.isfinite(lp).all(), "Nonfinite reference log probabilities"
            targets = ids[i, start + 1:end + 1].to(args.device)
            full_nll.extend((-lp.gather(1, targets[:, None]).squeeze(1)).cpu().tolist())
            for p in w["kl_positions"]:
                if start <= p < end:
                    selected[p] = lp[p - start].cpu().numpy()
        positions = w["kl_positions"]
        np.save(out / f"window-{i:03d}-logprobs.npy", np.stack([selected[p] for p in positions]))
        np.save(out / f"window-{i:03d}-nll.npy", np.asarray(full_nll, dtype=np.float32))
        rows.append({"window": i, "name": w["name"], "domain": w["domain"],
                     "positions": len(full_nll), "nll_mean": float(np.mean(full_nll)),
                     "perplexity": float(np.exp(np.mean(full_nll)))})
        nlls.extend(full_nll)
    summary = {"kind": "unquantized_transformers_layer_stream", "dtype": args.dtype,
               "model_path": args.model, "panel_sha256": hashlib.sha256(panel_bytes).hexdigest(),
               "vocab_size": cfg.vocab_size, "windows": rows,
               "positions": len(nlls), "nll_mean": float(np.mean(nlls)),
               "perplexity": float(np.exp(np.mean(nlls))), "seconds": time.monotonic() - started,
               "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
               "peak_xpu_gib": torch.xpu.max_memory_allocated() / 2**30 if args.device.startswith("xpu") else None,
               "torch": torch.__version__, "no_vision_no_mtp": True}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--panel", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--device", default="xpu")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--windows", type=int, default=0)
    run(p.parse_args())

#!/usr/bin/env python3
"""Offline speed and quality screen for an INT4/G128 target head."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open
import vllm._custom_ops  # noqa: F401
from vllm.model_executor.models.b70_draft_lmhead_int4 import (
    int4_lmhead_logits,
    quantize_lmhead_to_int4,
)


def measure(call, repeats=31, warmup=5):
    for _ in range(warmup):
        call()
    torch.xpu.synchronize()
    values = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        call()
        torch.xpu.synchronize()
        values.append((time.perf_counter_ns() - start) / 1e6)
    return {"median_ms": statistics.median(values), "min_ms": min(values),
            "samples_ms": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", required=True)
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with safe_open(args.weight, framework="pt", device="cpu") as handle:
        cpu_weight = handle.get_tensor("lm_head.weight")
    weight = cpu_weight.to("xpu")
    del cpu_weight
    qweight, scales, qzeros, group_size = quantize_lmhead_to_int4(weight)
    captured = torch.load(args.hidden_states, map_location="cpu", weights_only=True)
    hidden = captured["hidden_states"] if isinstance(captured, dict) else captured
    hidden = torch.unique(hidden.reshape(-1, weight.shape[1]), dim=0).half().to("xpu")

    result = {
        "status": "complete", "unique_hidden_rows": hidden.shape[0],
        "weight_shape": list(weight.shape), "arms": [],
    }
    for rows in sorted(set((1, min(4, hidden.shape[0]), hidden.shape[0]))):
        x = hidden[:rows].contiguous()
        fp16 = torch.nn.functional.linear(x, weight)
        int4 = int4_lmhead_logits(x, qweight, scales, qzeros, group_size)
        fp_top20 = fp16.topk(20, dim=1).indices
        int_top20 = int4.topk(20, dim=1).indices
        overlap = sum(len(set(a.tolist()) & set(b.tolist()))
                      for a, b in zip(fp_top20.cpu(), int_top20.cpu()))
        fp_logp = torch.log_softmax(fp16.float(), dim=1)
        int_logp = torch.log_softmax(int4.float(), dim=1)
        fp_prob = fp_logp.exp()
        kl = (fp_prob * (fp_logp - int_logp)).sum(1)
        delta = (int4.float() - fp16.float()).abs()
        quality = {
            "top1_agreement": float((fp16.argmax(1) == int4.argmax(1)).float().mean().item()),
            "top20_recall": overlap / (rows * 20),
            "mean_abs_logit_error": delta.mean().item(),
            "max_abs_logit_error": delta.max().item(),
            "mean_kl_fp16_to_int4": kl.mean().item(),
            "max_kl_fp16_to_int4": kl.max().item(),
        }
        fp_timing = measure(lambda: torch.nn.functional.linear(x, weight).argmax(1))
        int_timing = measure(lambda: int4_lmhead_logits(
            x, qweight, scales, qzeros, group_size).argmax(1))
        result["arms"].append({"rows": rows, "quality": quality,
                               "fp16": fp_timing, "int4": int_timing,
                               "speedup_pct": (fp_timing["median_ms"] /
                                               int_timing["median_ms"] - 1) * 100})
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

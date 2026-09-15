#!/usr/bin/env python3
"""Xe2 offline gate for an INT2 draft-head shortlist plus exact rerank.

This is deliberately not a serving patch. It compares the complete candidate
selection path with the deployed INT4/G128 draft head and records whether a
later serving implementation is justified.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl
from safetensors import safe_open

# Registers the Intel mixed-precision operators and exposes the deployed head
# packing helper from the production image.
import vllm._custom_ops  # noqa: F401
from vllm.model_executor.models.b70_draft_lmhead_int4 import (
    int4_lmhead_logits,
    quantize_lmhead_to_int4,
)


GROUP = 128
BLOCK_N = 64
KCAND = 8


@triton.jit
def emit_candidates(acc, mask_n, bm, bi, offs_m, pid, nblk,
                    KC: tl.constexpr, BN: tl.constexpr):
    masked = tl.where(mask_n[None, :], acc, float("-inf"))
    for c in tl.static_range(KC):
        mx = tl.max(masked, axis=1)
        am = tl.argmax(masked, axis=1)
        tl.store(bm + offs_m * (nblk * KC) + pid * KC + c, mx)
        tl.store(bi + offs_m * (nblk * KC) + pid * KC + c,
                 (pid * BN + am).to(tl.int32))
        masked = tl.where(tl.arange(0, BN)[None, :] == am[:, None],
                          float("-inf"), masked)


@triton.jit
def int2_shortlist_kernel(x, xsum, wq, scale4, zscale4, bm, bi,
                          K: tl.constexpr, n, stride_wq, stride_s,
                          stride_xsum, nblk, KC: tl.constexpr,
                          G: tl.constexpr, BM: tl.constexpr,
                          BN: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, G)
    mask_n = offs_n < n
    Q: tl.constexpr = K // 4
    NG: tl.constexpr = Q // G
    acc = tl.zeros((BM, BN), tl.float32)
    for g in range(0, NG):
        packed = tl.load(wq + offs_n[None, :] * stride_wq
                         + (g * G + offs_k)[:, None],
                         mask=mask_n[None, :], other=0).to(tl.uint16)
        for quarter in tl.static_range(4):
            bits = (packed >> (2 * quarter)) & 3
            # Exact FP16 bit pattern for 1 + q/4, q in [0,3].
            wv = (0x3C00 | (bits << 8)).to(tl.float16, bitcast=True)
            xv = tl.load(x + offs_m[:, None] * K
                         + (quarter * Q + g * G + offs_k)[None, :])
            group_index = quarter * NG + g
            sx = tl.load(xsum + offs_m * stride_xsum + group_index)
            sc = tl.load(scale4 + offs_n * stride_s + group_index,
                         mask=mask_n, other=0.0)
            zsc = tl.load(zscale4 + offs_n * stride_s + group_index,
                          mask=mask_n, other=0.0)
            acc += tl.dot(xv, wv) * sc[None, :]
            acc -= sx[:, None] * zsc[None, :]
    emit_candidates(acc, mask_n, bm, bi, offs_m, pid, nblk, KC, BN)


@triton.jit
def rerank_fp16_kernel(x, weight, indices, scores, K: tl.constexpr,
                       stride_w, R: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0)
    j = tl.program_id(1)
    token = tl.load(indices + m * R + j)
    acc = tl.zeros((BK,), tl.float32)
    for k0 in range(0, K, BK):
        offsets = k0 + tl.arange(0, BK)
        acc += (tl.load(x + m * K + offsets).to(tl.float32)
                * tl.load(weight + token * stride_w + offsets).to(tl.float32))
    tl.store(scores + m * R + j, tl.sum(acc, axis=0))


def quantize_int2(weight: torch.Tensor):
    n, k = weight.shape
    levels = 3
    packed = torch.empty((n, k // 4), dtype=torch.uint8, device="xpu")
    scale4 = torch.empty((n, k // GROUP), dtype=torch.float16, device="xpu")
    zscale4 = torch.empty_like(scale4)
    for start in range(0, n, 4096):
        end = min(n, start + 4096)
        grouped = weight[start:end].float().reshape(end - start, k // GROUP, GROUP)
        low = grouped.amin(2)
        high = grouped.amax(2)
        scale = ((high - low) / levels).clamp(min=1e-8)
        zero = (-low / scale).round().clamp(0, levels)
        q = (grouped / scale[..., None] + zero[..., None]).round()
        q = q.clamp(0, levels).to(torch.uint8).reshape(end - start, k)
        quarter = k // 4
        packed[start:end] = (q[:, :quarter] | (q[:, quarter:2 * quarter] << 2)
                             | (q[:, 2 * quarter:3 * quarter] << 4)
                             | (q[:, 3 * quarter:] << 6))
        scale4[start:end] = (4 * scale).half()
        zscale4[start:end] = (4 * scale + zero * scale).half()
        del grouped, low, high, scale, zero, q
    torch.xpu.synchronize()
    return packed, scale4, zscale4


def pad_rows(x: torch.Tensor):
    rows = max(16, 1 << (x.shape[0] - 1).bit_length())
    if rows == x.shape[0]:
        return x
    return torch.cat((x, x.new_zeros((rows - x.shape[0], x.shape[1]))))


def shortlist(x, weight, packed, scale4, zscale4, rerank):
    real_rows = x.shape[0]
    x = pad_rows(x.contiguous())
    rows, k = x.shape
    n = weight.shape[0]
    nblk = (n + BLOCK_N - 1) // BLOCK_N
    xsum = x.reshape(rows, k // GROUP, GROUP).float().sum(2).contiguous()
    bm = torch.empty((rows, nblk * KCAND), dtype=torch.float32, device="xpu")
    bi = torch.empty((rows, nblk * KCAND), dtype=torch.int32, device="xpu")
    warps = 2 if rows <= 16 else 4 if rows <= 48 else 8
    int2_shortlist_kernel[(nblk,)](
        x, xsum, packed, scale4, zscale4, bm, bi, k, n,
        packed.stride(0), scale4.stride(0), xsum.stride(0), nblk,
        KCAND, G=GROUP, BM=rows, BN=BLOCK_N, num_warps=warps, num_stages=1)
    top = bm.topk(rerank, dim=1).indices
    indices = bi.gather(1, top).contiguous()
    scores = torch.empty((rows, rerank), dtype=torch.float32, device="xpu")
    rerank_fp16_kernel[(rows, rerank)](
        x, weight, indices, scores, k, weight.stride(0), R=rerank, BK=512,
        num_warps=4)
    winner = indices.gather(1, scores.argmax(1, keepdim=True)).squeeze(1)
    return winner[:real_rows]


def measure(call, repeats=31, warmup=5):
    for _ in range(warmup):
        call()
    torch.xpu.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        call()
        torch.xpu.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {"median_ms": statistics.median(samples),
            "min_ms": min(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-states")
    parser.add_argument("--rows", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(260915)
    with safe_open(args.weight, framework="pt", device="cpu") as handle:
        cpu_weight = handle.get_tensor("lm_head.weight")
    weight = cpu_weight.to("xpu")
    del cpu_weight
    qweight, scales, qzeros, group_size = quantize_lmhead_to_int4(weight)
    packed, scale4, zscale4 = quantize_int2(weight)

    if args.hidden_states:
        captured = torch.load(args.hidden_states, map_location="cpu", weights_only=True)
        hidden = captured["hidden_states"] if isinstance(captured, dict) else captured
        hidden = torch.unique(hidden.reshape(-1, weight.shape[1]), dim=0)
        hidden = hidden[:args.rows].half().to("xpu")
        source = "captured"
    else:
        hidden = torch.randn((args.rows, weight.shape[1]), dtype=torch.float16,
                             device="xpu") * 0.1
        source = "synthetic-normal-0.1"

    result = {
        "status": "complete", "source": source,
        "weight": {"shape": list(weight.shape), "dtype": str(weight.dtype)},
        "bytes": {
            "fp16": weight.numel() * weight.element_size(),
            "int4": qweight.numel() * qweight.element_size()
                    + scales.numel() * scales.element_size(),
            "int2": packed.numel() + scale4.numel() * scale4.element_size()
                    + zscale4.numel() * zscale4.element_size(),
        },
        "arms": [],
    }
    result["unique_hidden_rows"] = hidden.shape[0]
    for rows in (1, 4, min(16, hidden.shape[0])):
        x = hidden[:rows].contiguous()
        reference = int4_lmhead_logits(x, qweight, scales, qzeros, group_size).argmax(1)
        for rerank in (32, 64, 128):
            selected = shortlist(x, weight, packed, scale4, zscale4, rerank)
            correct = int((selected == reference).sum().item())
            timing = measure(lambda: shortlist(x, weight, packed, scale4, zscale4,
                                               rerank))
            result["arms"].append({"rows": rows, "rerank": rerank,
                                   "int4_top1_recall": correct / rows,
                                   "matches": correct, **timing})
        baseline = measure(lambda: int4_lmhead_logits(
            x, qweight, scales, qzeros, group_size).argmax(1))
        result["arms"].append({"rows": rows, "variant": "int4-full-argmax",
                               **baseline})
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

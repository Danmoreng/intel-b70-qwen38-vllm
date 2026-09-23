#!/usr/bin/env python3
"""Paired Q128/native XPU attention timings at fixed new-token count."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import statistics

import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func as native


BLOCK = 1664
HEADS_Q = 24
HEADS_KV = 4
DIM = 256
LENGTHS = (16 * 1024, 64 * 1024, 128 * 1024, 192 * 1024)


def measure(fn):
    start = torch.xpu.Event(enable_timing=True)
    stop = torch.xpu.Event(enable_timing=True)
    start.record()
    fn()
    stop.record()
    torch.xpu.synchronize()
    return start.elapsed_time(stop)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--q-len", type=int, default=512)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--descale", type=float, default=0.01)
    parser.add_argument("--kv-lengths", type=int, nargs="+", default=LENGTHS)
    args = parser.parse_args()
    assert args.q_len >= 256 and args.q_len <= 6656
    assert args.pairs >= 2
    assert args.kv_lengths and all(length > 0 for length in args.kv_lengths)
    torch.manual_seed(38)
    torch.ops.load_library("/opt/b70/tiles.so")
    available_before, total = torch.xpu.mem_get_info()
    max_blocks = (max(args.kv_lengths) + BLOCK - 1) // BLOCK
    q = torch.randn((args.q_len, HEADS_Q, DIM), device="xpu", dtype=torch.float16)
    kv = torch.empty((max_blocks, BLOCK, HEADS_KV, 2, DIM), device="xpu",
                     dtype=torch.float8_e4m3fn)
    for offset in range(0, max_blocks, 8):
        count = min(8, max_blocks - offset)
        shape = (count, BLOCK, HEADS_KV, 2, DIM)
        kv[offset:offset + count].copy_(
            (torch.randn(shape, device="xpu", dtype=torch.float16) * 60).to(torch.float8_e4m3fn)
        )
    k, v = kv[:, :, :, 0, :], kv[:, :, :, 1, :]
    assert k.stride() == v.stride() == (3407872, 2048, 512, 1)
    block_table = torch.arange(max_blocks, device="xpu", dtype=torch.int32).reshape(1, -1)
    cu_q = torch.tensor([0, args.q_len], device="xpu", dtype=torch.int32)
    used_k = torch.tensor([0], device="xpu", dtype=torch.int32)
    descale = torch.full((1,), args.descale, device="xpu", dtype=torch.float32)
    torch.xpu.synchronize()
    available_after, _ = torch.xpu.mem_get_info()
    rows = []
    for kv_len in args.kv_lengths:
        used_k.fill_(kv_len)
        torch.xpu.synchronize()

        def q128():
            return torch.ops.b70_tiles.forward(
                q, k, v, block_table, cu_q, used_k, descale, descale, kv_len, 1
            )

        def reference():
            return native(
                q, k, v, max_seqlen_q=args.q_len, cu_seqlens_q=cu_q,
                max_seqlen_k=kv_len, seqused_k=used_k,
                k_descale=descale, v_descale=descale, block_table=block_table,
                causal=True, softmax_scale=0.0625,
            )

        for _ in range(3):
            q128()
            reference()
        torch.xpu.synchronize()
        y = q128().cpu()
        torch.xpu.empty_cache()
        ref = reference().cpu()
        torch.xpu.synchronize()
        delta = (y.float() - ref.float()).abs()
        correctness = {
            "allclose": bool(torch.allclose(y, ref, rtol=0.01, atol=0.002)),
            "finite": bool(y.isfinite().all().item()),
            "max_abs": float(delta.max().item()),
            "mean_abs": float(delta.mean().item()),
            "output_max_abs": float(y.float().abs().max().item()),
            "output_rms": float(y.float().square().mean().sqrt().item()),
        }
        if not correctness["allclose"] or not correctness["finite"]:
            raise RuntimeError(f"operator mismatch at {kv_len}: {correctness}")
        del y, ref, delta
        torch.xpu.synchronize()
        times = {"q128": [], "native": []}
        # ABBA / BAAB pairing reduces monotonic temperature/clock drift.
        for pair in range(args.pairs):
            order = ("q128", "native", "native", "q128") if pair % 2 == 0 else (
                "native", "q128", "q128", "native"
            )
            for arm in order:
                times[arm].append(measure(q128 if arm == "q128" else reference))
        row = {
            "kv_tokens": kv_len,
            "q_tokens": args.q_len,
            "blocks_allocated": max_blocks,
            "blocks_used": (kv_len + BLOCK - 1) // BLOCK,
            "correctness": correctness,
            "timings_ms": times,
            "median_ms": {name: statistics.median(samples) for name, samples in times.items()},
        }
        row["q128_speedup_percent"] = 100 * (
            row["median_ms"]["native"] / row["median_ms"]["q128"] - 1
        )
        rows.append(row)
        print(json.dumps({"kv_tokens": kv_len, "median_ms": row["median_ms"],
                          "q128_speedup_percent": row["q128_speedup_percent"],
                          "correctness": correctness}), flush=True)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "gpu_name": torch.xpu.get_device_name(0),
        "q128_binary_sha256": hashlib.sha256(open("/opt/b70/tiles.so", "rb").read()).hexdigest(),
        "available_gpu_bytes_before": available_before,
        "available_gpu_bytes_after_alloc": available_after,
        "total_gpu_bytes": total,
        "block_size": BLOCK,
        "kv_stride": list(k.stride()),
        "q_len": args.q_len,
        "descale": args.descale,
        "pairs": args.pairs,
        "kv_lengths": args.kv_lengths,
        "rows": rows,
    }
    with open(args.output, "x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()

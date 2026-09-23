#!/usr/bin/env python3
"""Isolated Q128 and M04 device times with production-stride interleaved FP8 KV."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func as native


PAGE, HQ, HKV, D = 1664, 24, 4, 256


def timed(fn):
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", required=True, type=Path)
    parser.add_argument("--m04", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(38)
    torch.ops.load_library(str(args.tiles))
    torch.ops.load_library(str(args.m04))
    free_before, total = torch.xpu.mem_get_info()
    contexts = (65536, 196608)
    blocks = (max(contexts) + PAGE - 1) // PAGE
    kv = torch.empty((blocks, PAGE, HKV, 2, D), device="xpu", dtype=torch.float8_e4m3fn)
    for start in range(0, blocks, 8):
        count = min(8, blocks - start)
        shape = (count, PAGE, HKV, 2, D)
        kv[start:start + count].copy_(
            (torch.randn(shape, device="xpu", dtype=torch.float16) * 60).to(torch.float8_e4m3fn)
        )
    k, v = kv[:, :, :, 0, :], kv[:, :, :, 1, :]
    assert k.stride() == v.stride() == (3407872, 2048, 512, 1)
    bt = torch.arange(blocks, device="xpu", dtype=torch.int32).reshape(1, -1)
    scale = torch.tensor([0.01], device="xpu", dtype=torch.float32)
    q128_q = torch.randn((6656, HQ, D), device="xpu", dtype=torch.float16)
    q128_cu = torch.tensor([0, 6656], device="xpu", dtype=torch.int32)
    m04_q = torch.randn((4, HQ, D), device="xpu", dtype=torch.float16)
    packed_q = m04_q.view(4, HKV, 6, D).permute(1, 0, 2, 3).reshape(1, 4 * HQ, D).contiguous()
    m04_cu = torch.tensor([0, 4], device="xpu", dtype=torch.int32)
    m04_out = torch.empty_like(packed_q)
    splits = 16
    temp = torch.empty((1, 4 * HQ * splits, D), device="xpu", dtype=torch.float16)
    sums = torch.empty((1, 4 * HQ, splits), device="xpu", dtype=torch.float32)
    maxima = torch.empty_like(sums)
    used = torch.tensor([0], device="xpu", dtype=torch.int32)
    rows = []
    for context in contexts:
        used.fill_(context)
        torch.xpu.synchronize()

        def q128():
            return torch.ops.b70_tiles.forward(
                q128_q, k, v, bt, q128_cu, used, scale, scale, context, 1
            )

        def m04():
            torch.ops.b70_ops.shared_kv_verify_out(
                packed_q, k, v, bt, m04_cu, used, scale, scale,
                m04_out, temp, sums, maxima, 200704, splits, 8,
            )
            return m04_out

        checks = {}
        for name, fn, q, cu in (
            ("q128", q128, q128_q, q128_cu),
            ("m04", m04, m04_q, m04_cu),
        ):
            for _ in range(3):
                fn()
            y = fn().cpu()
            if name == "m04":
                y = y.view(HKV, 4, 6, D).permute(1, 0, 2, 3).reshape(4, HQ, D).contiguous()
            torch.xpu.empty_cache()
            reference = native(
                q, k, v, max_seqlen_q=q.shape[0], cu_seqlens_q=cu,
                max_seqlen_k=context, seqused_k=used,
                k_descale=scale, v_descale=scale, block_table=bt,
                causal=True, softmax_scale=0.0625,
            ).cpu()
            delta = (y.float() - reference.float()).abs()
            checks[name] = {
                "allclose": bool(torch.allclose(y, reference, rtol=0.01, atol=0.002)),
                "finite": bool(y.isfinite().all()),
                "max_abs": float(delta.max()),
                "output_max_abs": float(y.float().abs().max()),
            }
            if not checks[name]["allclose"] or not checks[name]["finite"]:
                raise RuntimeError(f"{name} mismatch at {context}: {checks[name]}")
            del y, reference, delta
            torch.xpu.empty_cache()
        samples = {"q128": [], "m04": []}
        for i in range(args.samples):
            for name in (("q128", "m04") if i % 2 == 0 else ("m04", "q128")):
                samples[name].append(timed(q128 if name == "q128" else m04))
        row = {
            "kv_tokens": context,
            "checks": checks,
            "samples_ms": samples,
            "median_ms": {name: statistics.median(values) for name, values in samples.items()},
        }
        rows.append(row)
        print(json.dumps({"label": args.label, "kv_tokens": context,
                          "median_ms": row["median_ms"], "checks": checks}), flush=True)
    result = {
        "label": args.label,
        "torch_version": torch.__version__,
        "gpu_name": torch.xpu.get_device_name(0),
        "free_gpu_bytes_before": free_before,
        "total_gpu_bytes": total,
        "tiles_sha256": hashlib.sha256(args.tiles.read_bytes()).hexdigest(),
        "m04_sha256": hashlib.sha256(args.m04.read_bytes()).hexdigest(),
        "kv_stride": list(k.stride()),
        "samples_per_shape": args.samples,
        "rows": rows,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()

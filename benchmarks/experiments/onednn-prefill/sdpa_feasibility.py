#!/usr/bin/env python3
"""Small same-operand screen for fused FP16 SDPA on the pinned XPU stack.

This is an operator feasibility screen, not the proposed native adapter. It
includes the FP8 gather, FP16 conversion, query packing, and output unpacking.
"""

import argparse
import json
import os
import statistics

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right
from torch.profiler import ProfilerActivity, profile


BLOCK = 1664
HEADS = 24
KV_HEADS = 4
DIM = 256
SCALE = DIM ** -0.5


def timed(fn):
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    value = fn()
    end.record()
    end.synchronize()
    return value, start.elapsed_time(end)


def run(query_rows, length, repeats, inspect_kernel, native, adapter, native_gather, memory):
    torch.manual_seed(38)
    torch.ops.load_library("/opt/b70/tiles.so")
    if native and not adapter:
        torch.ops.load_library(os.environ.get("B70_ONEDNN_LIB", "/probe/native_sdpa.so"))
    pages_needed = (length + BLOCK - 1) // BLOCK
    page_count = pages_needed + 2
    q = torch.randn((query_rows, HEADS, DIM), device="xpu", dtype=torch.float16) * .1
    cache = torch.randn((page_count, BLOCK, KV_HEADS, 2, DIM), device="xpu", dtype=torch.float16)
    cache = (cache * .1).to(torch.float8_e4m3fn)
    k, v = cache[:, :, :, 0, :], cache[:, :, :, 1, :]
    assert k.stride() == (3407872, 2048, 512, 1)
    pages = torch.randperm(page_count, device="xpu", dtype=torch.int32)[:pages_needed]
    table = pages.reshape(1, -1).contiguous()
    cu_q = torch.tensor([0, query_rows], device="xpu", dtype=torch.int32)
    used = torch.tensor([length], device="xpu", dtype=torch.int32)
    ks = torch.tensor([.75], device="xpu", dtype=torch.float32)
    vs = torch.tensor([1.25], device="xpu", dtype=torch.float32)
    mask = causal_lower_right(query_rows, length)
    divisor = torch.tensor([16.0], device="xpu", dtype=torch.float16)
    negative_inf = torch.tensor([float("-inf")], device="xpu", dtype=torch.float32)
    if native_gather:
        key_buffer = torch.empty((length, DIM), device="xpu", dtype=torch.float16)
        value_buffer = torch.empty_like(key_buffer)
        query_buffer = torch.empty((6, query_rows, DIM), device="xpu", dtype=torch.float16)
        result_buffer = torch.empty_like(query_buffer)

    def baseline():
        return torch.ops.b70_tiles.forward(q, k, v, table, cu_q, used, ks, vs, length, 1)

    def candidate():
        if adapter:
            import b70_attention
            return b70_attention.flash_attn_varlen_func(
                q=q, k=k, v=v, max_seqlen_q=query_rows, cu_seqlens_q=cu_q,
                max_seqlen_k=length, seqused_k=used, k_descale=ks,
                v_descale=vs, block_table=table, causal=True,
                softmax_scale=SCALE)
        result = torch.empty_like(q)
        for h in range(KV_HEADS):
            if native_gather:
                torch.ops.b70_sdpa_probe.gather_dequant(k, pages, ks, key_buffer, h)
                torch.ops.b70_sdpa_probe.gather_dequant(v, pages, vs, value_buffer, h)
                query_buffer.copy_(q[:, h * 6:(h + 1) * 6, :].permute(1, 0, 2))
                key, val = key_buffer, value_buffer
                queries = query_buffer[None]
            else:
                key = (k[:, :, h].index_select(0, pages.long()).flatten(0, 1)[:length].to(torch.float16) * ks).to(torch.float16)
                val = (v[:, :, h].index_select(0, pages.long()).flatten(0, 1)[:length].to(torch.float16) * vs).to(torch.float16)
                queries = q[:, h * 6:(h + 1) * 6, :].permute(1, 0, 2).unsqueeze(0).contiguous()
            if native:
                output = result_buffer if native_gather else torch.empty_like(queries[0])
                torch.ops.b70_sdpa_probe.forward(
                    queries[0], key[None], val[None], output, divisor, negative_inf)
                output = output[None]
            else:
                output = F.scaled_dot_product_attention(
                    queries, key[None, None], val[None, None], attn_mask=mask, scale=SCALE)
            result[:, h * 6:(h + 1) * 6, :].copy_(output[0].permute(1, 0, 2))
        return result

    reference, _ = timed(baseline)
    output, first_ms = timed(candidate)
    delta = (output.float() - reference.float()).abs()
    correctness = {
        "finite": bool(torch.isfinite(output).all().item()),
        "allclose": bool(torch.allclose(output, reference, rtol=.01, atol=.002)),
        "max_abs": float(delta.max().item()),
        "relative_l2": float((delta.norm() / reference.float().norm()).item()),
    }
    if not correctness["finite"] or not correctness["allclose"]:
        raise RuntimeError(json.dumps(correctness))
    candidate_name = ("adapter" if adapter else
                      "native_gather_graph_complete" if native_gather else
                      "native_graph_complete" if native else "torch_sdpa_complete")
    samples = {"q128": [], candidate_name: []}
    for _ in range(repeats):
        for name, fn in (("q128", baseline), (candidate_name, candidate)):
            _, elapsed = timed(fn)
            samples[name].append(elapsed)
    if inspect_kernel:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
            candidate()
            torch.xpu.synchronize()
        print(json.dumps({"kernels": [
            {"name": event.key, "count": event.count, "xpu_us": event.device_time_total}
            for event in sorted(prof.key_averages(), key=lambda event: event.device_time_total, reverse=True)
            if event.device_time_total > 0
        ][:20]}))
    memory_result = None
    if memory:
        memory_result = {}
        for name, fn in (("q128", baseline), (candidate_name, candidate)):
            torch.xpu.synchronize()
            torch.xpu.reset_peak_memory_stats()
            before = torch.xpu.memory_allocated()
            transient_output = fn()
            torch.xpu.synchronize()
            memory_result[name] = {
                "allocated_before_bytes": before,
                "peak_extra_allocated_bytes": torch.xpu.max_memory_allocated() - before,
                "reserved_after_bytes": torch.xpu.memory_reserved(),
            }
            del transient_output
    print(json.dumps({
        "q": query_rows, "l": length, "first_candidate_ms": first_ms,
        "correctness": correctness,
        "median_ms": {name: statistics.median(values) for name, values in samples.items()},
        "samples_ms": samples,
        "memory": memory_result,
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=int, default=256)
    parser.add_argument("--l", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--inspect-kernel", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--matrix", action="store_true",
                        help="test nonmultiple Q and 1664-page boundaries in one process")
    parser.add_argument("--adapter", action="store_true")
    parser.add_argument("--native-gather", action="store_true")
    parser.add_argument("--memory", action="store_true")
    args = parser.parse_args()
    if args.native_gather and not args.native:
        parser.error("--native-gather requires --native")
    if args.matrix:
        for q, length in ((257, 1663), (257, 1664), (257, 1665),
                          (513, 3327), (513, 3328), (513, 3329)):
            run(q, length, args.repeats, False, args.native, args.adapter, args.native_gather, args.memory)
    else:
        run(args.q, args.l, args.repeats, args.inspect_kernel, args.native, args.adapter, args.native_gather, args.memory)

"""M04 custom split-K shared-KV verification kernel gate."""
from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

import torch
from vllm_xpu_kernels import _vllm_fa2_C  # noqa: F401
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func


ROOT = Path("/work")
PAGE, HQ, HKV, HEAD = 1664, 24, 4, 256
SCALE = HEAD**-0.5
QLENS = (2, 3, 5)
CONTEXTS = (8192, 65536, 131072, 196608)
SPLITS = (4, 8, 16, 24, 32)

torch.ops.load_library(str(ROOT / "b70_ops.so"))
torch.manual_seed(405)


def save(name: str, value: object) -> None:
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n")


def timed(function):
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    value = function()
    end.record()
    end.synchronize()
    return value, {"event_ms": start.elapsed_time(end),
                   "wall_ms": (time.perf_counter() - wall) * 1000}


def delta(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    error = (actual.float() - expected.float()).abs()
    return {"finite": bool(actual.isfinite().all().cpu()),
            "allclose": bool(torch.allclose(actual.float(), expected.float(),
                                             rtol=0.01, atol=0.003)),
            "max_abs": float(error.max().cpu()),
            "rms": float(error.square().mean().sqrt().cpu())}


max_pages = (max(CONTEXTS) + PAGE - 1) // PAGE
base = torch.randn((max_pages, PAGE, HKV, 2, HEAD), dtype=torch.float16,
                   device="xpu").to(torch.float8_e4m3fn)
k, v = base[:, :, :, 0, :], base[:, :, :, 1, :]
assert list(k.stride()) == [PAGE * HKV * 2 * HEAD, HKV * 2 * HEAD,
                            2 * HEAD, 1]
physical = torch.randperm(max_pages, device="xpu", dtype=torch.int64).int()
k_descale = torch.tensor([0.75], device="xpu", dtype=torch.float32)
v_descale = torch.tensor([1.25], device="xpu", dtype=torch.float32)


def make_case(q_len: int, kv_len: int, num_splits: int):
    q = torch.randn((q_len, HQ, HEAD), device="xpu", dtype=torch.float16)
    pages = (kv_len + PAGE - 1) // PAGE
    block_table = physical[:pages][None, :].contiguous()
    cu_q = torch.tensor([0, q_len], device="xpu", dtype=torch.int32)
    used = torch.tensor([kv_len], device="xpu", dtype=torch.int32)
    out_control = torch.empty_like(q)
    packed_heads = q_len * HQ
    packed_out = torch.empty((1, packed_heads, HEAD), device="xpu",
                             dtype=torch.float16)
    temp = torch.empty((1, packed_heads * num_splits, HEAD), device="xpu",
                       dtype=torch.float16)
    sums = torch.empty((1, packed_heads, num_splits), device="xpu",
                       dtype=torch.float32)
    maxima = torch.empty_like(sums)

    def control():
        return flash_attn_varlen_func(
            q, k, v, max_seqlen_q=q_len, cu_seqlens_q=cu_q,
            max_seqlen_k=kv_len, seqused_k=used, block_table=block_table,
            softmax_scale=SCALE, causal=True, k_descale=k_descale,
            v_descale=v_descale, out=out_control)

    def challenger():
        packed_q = (q.view(q_len, HKV, HQ // HKV, HEAD)
                    .permute(1, 0, 2, 3).reshape(1, packed_heads, HEAD)
                    .contiguous())
        torch.ops.b70_ops.shared_kv_verify_out(
            packed_q, k, v, block_table, cu_q, used, k_descale,
            v_descale, packed_out, temp, sums, maxima, kv_len, num_splits, 8)
        return (packed_out.view(HKV, q_len, HQ // HKV, HEAD)
                .permute(1, 0, 2, 3).reshape(q_len, HQ, HEAD).contiguous())

    return q, block_table, control, challenger


checks = []
for q_len in QLENS:
    for kv_len in (65, PAGE - 1, PAGE, PAGE + 1, 8192, 196608):
        _, _, control, _ = make_case(q_len, kv_len, 1)
        expected = control()
        for num_splits in SPLITS:
            q, block_table, _, challenger = make_case(
                q_len, kv_len, num_splits)
            # make_case creates a new Q, so obtain the matching control.
            q, block_table, matching_control, challenger = make_case(
                q_len, kv_len, num_splits)
            expected = matching_control()
            actual = challenger()
            torch.xpu.synchronize()
            comparison = delta(actual, expected)
            check = {"q_len": q_len, "kv_len": kv_len,
                     "num_splits": num_splits,
                     "challenger_vs_deployed": comparison}
            checks.append(check)
            save("split-correctness.json", checks)
            print("check", check, flush=True)

assert all(c["challenger_vs_deployed"]["finite"] and
           c["challenger_vs_deployed"]["allclose"] for c in checks), \
    "M04 split-K correctness gate failed"

rows = []
for q_len in QLENS:
    for kv_len in CONTEXTS:
        _, _, control, _ = make_case(q_len, kv_len, 1)
        functions = {"deployed_split_k": control}
        for num_splits in SPLITS:
            _, _, _, challenger = make_case(q_len, kv_len, num_splits)
            functions[f"shared_kv_split_{num_splits}"] = challenger
        for function in functions.values():
            for _ in range(5):
                function()
        torch.xpu.synchronize()
        samples = {name: [] for name in functions}
        rng = random.Random(405 + q_len + kv_len)
        for _ in range(21):
            names = list(functions)
            rng.shuffle(names)
            for name in names:
                _, sample = timed(functions[name])
                samples[name].append(sample)
        medians = {name: statistics.median(s["event_ms"] for s in values)
                   for name, values in samples.items()}
        best_name = min((name for name in medians if name != "deployed_split_k"),
                        key=medians.get)
        row = {"q_len": q_len, "kv_len": kv_len,
               "median_event_ms": medians, "best_challenger": best_name,
               "best_speedup_pct": 100 * (medians["deployed_split_k"] /
                                             medians[best_name] - 1),
               "samples": samples}
        rows.append(row)
        save("split-results.json", {
            "scope": "M04 packed-row shared-KV split-K attention, including pack/unpack",
            "rows": rows})
        print("timing", {key: value for key, value in row.items()
                         if key != "samples"}, flush=True)

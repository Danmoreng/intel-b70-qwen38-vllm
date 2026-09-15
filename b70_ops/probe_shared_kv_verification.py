"""M04 gate: split-K decode versus shared-KV chunk verification attention."""
from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

import torch
from vllm_xpu_kernels import _vllm_fa2_C  # noqa: F401 - registers the op
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func


ROOT = Path("/work")
PAGE = 1664
HQ = 24
HKV = 4
HEAD = 256
SCALE = HEAD**-0.5
QLENS = (2, 3, 5)
CONTEXTS = (8192, 65536, 131072, 196608)

torch.manual_seed(404)


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
    return {
        "finite": bool(actual.isfinite().all().cpu()),
        "allclose": bool(torch.allclose(actual.float(), expected.float(),
                                         rtol=0.01, atol=0.003)),
        "max_abs": float(error.max().cpu()),
        "rms": float(error.square().mean().sqrt().cpu()),
    }


max_pages = (max(CONTEXTS) + PAGE - 1) // PAGE
# This interleaved allocation exactly reproduces production K/V strides.
base = torch.randn((max_pages, PAGE, HKV, 2, HEAD), dtype=torch.float16,
                   device="xpu").to(torch.float8_e4m3fn)
k = base[:, :, :, 0, :]
v = base[:, :, :, 1, :]
assert list(k.stride()) == [PAGE * HKV * 2 * HEAD, HKV * 2 * HEAD,
                            2 * HEAD, 1]
physical = torch.randperm(max_pages, device="xpu", dtype=torch.int64).int()
k_descale = torch.tensor([0.75], device="xpu", dtype=torch.float32)
v_descale = torch.tensor([1.25], device="xpu", dtype=torch.float32)


def make_case(q_len: int, kv_len: int):
    q = torch.randn((q_len, HQ, HEAD), device="xpu", dtype=torch.float16)
    pages = (kv_len + PAGE - 1) // PAGE
    shared_bt = physical[:pages][None, :].contiguous()
    cu_shared = torch.tensor([0, q_len], device="xpu", dtype=torch.int32)
    used_shared = torch.tensor([kv_len], device="xpu", dtype=torch.int32)
    out_control = torch.empty_like(q)
    out_shared = torch.empty_like(q)

    def control():
        # This is the deployed speculative-attention route. The Python shim
        # expands a q_len block into q_len independent one-row decode jobs.
        return flash_attn_varlen_func(
            q, k, v, max_seqlen_q=q_len, cu_seqlens_q=cu_shared,
            max_seqlen_k=kv_len, seqused_k=used_shared,
            block_table=shared_bt, softmax_scale=SCALE, causal=True,
            k_descale=k_descale, v_descale=v_descale, out=out_control)

    def shared_kv():
        # Call the native paged causal Q-block path directly. Bottom-right
        # causality gives row j exactly kv_len-q_len+j+1 visible KV tokens,
        # identical to the deployed pseudo-sequence transformation.
        result, _ = torch.ops._vllm_fa2_C.varlen_fwd(
            q, k, v, out_shared, cu_shared, torch.zeros_like(cu_shared),
            used_shared, None, shared_bt, None, q_len, kv_len, 0.0,
            k_descale, v_descale, SCALE, None, False, True, -1, -1, 0.0,
            False, None, None, False, None, None)
        return result

    return q, shared_bt, control, shared_kv


checks = []
# Boundaries cover tiny history, both sides of a physical page boundary, and
# the maximum context. q_len 2/3/5 covers short accepted/rejected MTP rounds.
for q_len in QLENS:
    for kv_len in (65, PAGE - 1, PAGE, PAGE + 1, 8192, 65536, 196608):
        q, bt, control, shared_kv = make_case(q_len, kv_len)
        expected = control()
        actual = shared_kv()
        torch.xpu.synchronize()
        comparison = delta(actual, expected)
        check = {"q_len": q_len, "kv_len": kv_len,
                 "shared_vs_deployed": comparison}

        # Independent dense FP32 oracle is intentionally limited to short
        # histories. It validates paging, FP8 descale, GQA and causal limits.
        if kv_len <= PAGE + 1:
            keys = k[bt[0].long()].reshape(-1, HKV, HEAD)[:kv_len].float()
            vals = v[bt[0].long()].reshape(-1, HKV, HEAD)[:kv_len].float()
            keys *= k_descale
            vals *= v_descale
            gold = torch.empty_like(q, dtype=torch.float32)
            positions = torch.arange(kv_len, device="xpu")
            for head in range(HQ):
                scores = q[:, head].float() @ keys[:, head // 6].T * SCALE
                visible = kv_len - q_len + torch.arange(
                    q_len, device="xpu")[:, None]
                scores.masked_fill_(positions[None, :] > visible,
                                    -float("inf"))
                gold[:, head] = scores.softmax(-1) @ vals[:, head // 6]
            check["shared_vs_fp32"] = delta(actual, gold)
        checks.append(check)
        save("correctness.json", checks)
        print("check", check, flush=True)

assert all(c["shared_vs_deployed"]["finite"] and
           c["shared_vs_deployed"]["allclose"] and
           ("shared_vs_fp32" not in c or c["shared_vs_fp32"]["allclose"])
           for c in checks), "M04 correctness gate failed"

rows = []
for q_len in QLENS:
    for kv_len in CONTEXTS:
        _, _, control, shared_kv = make_case(q_len, kv_len)
        functions = {"deployed_split_k": control,
                     "shared_kv_qblock": shared_kv}
        for function in functions.values():
            for _ in range(5):
                function()
        torch.xpu.synchronize()
        samples = {name: [] for name in functions}
        rng = random.Random(404 + q_len + kv_len)
        for _ in range(21):
            names = list(functions)
            rng.shuffle(names)
            for name in names:
                _, sample = timed(functions[name])
                samples[name].append(sample)
        medians = {name: statistics.median(s["event_ms"] for s in values)
                   for name, values in samples.items()}
        row = {
            "q_len": q_len,
            "kv_len": kv_len,
            "median_event_ms": medians,
            "shared_kv_speedup_pct":
                100 * (medians["deployed_split_k"] /
                       medians["shared_kv_qblock"] - 1),
            "samples": samples,
        }
        rows.append(row)
        save("results.json", {
            "scope": "isolated exact-shape FP8 paged verification attention",
            "rows": rows,
        })
        print("timing", {key: value for key, value in row.items()
                         if key != "samples"}, flush=True)

# One final trace records the actual kernel families selected by both arms.
_, _, control, shared_kv = make_case(5, 65536)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                       torch.profiler.ProfilerActivity.XPU],
                            record_shapes=True) as profiler:
    with torch.profiler.record_function("deployed_split_k"):
        control()
    with torch.profiler.record_function("shared_kv_qblock"):
        shared_kv()
torch.xpu.synchronize()
profiler.export_chrome_trace(str(ROOT / "dispatch-trace.json"))

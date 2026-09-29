"""Production B70 attention dispatch: Q128/M04, exact-length oneDNN, mixed rows."""

import os

import torch

from b70_attention_base import flash_attn_varlen_func as fallback
from b70_attention_base import q128_eligible


MIN_KV = 16384
MAX_KV = 196608
PAGE = 1664
QUALIFICATION_ROUTE_OFF = os.environ.get("B70_QUALIFICATION_ROUTE_OFF") == "1"
if QUALIFICATION_ROUTE_OFF and os.environ.get("B70_QUALIFICATION_CONTROL") != "1":
    raise RuntimeError("route-off control requires explicit offline qualification")
_COUNTERS = {name: {"calls": 0, "query_tokens": 0, "kv_tokens": 0} for name in (
    "onednn_single", "onednn_mixed", "base_single", "base_mixed",
    "capture", "unsupported_mixed", "empty_mixed")}


def _required_flag(name, value):
    if os.environ.get(name) != value:
        raise RuntimeError(f"production policy requires {name}={value}")


for _name, _value in (
    ("B70_ONEDNN_PREFILL", "1"),
    ("B70_ONEDNN_MIXED_ROUTE", "1"),
    ("B70_ONEDNN_MIN_KV", str(MIN_KV)),
    ("B70_ONEDNN_MAX_KV", str(MAX_KV)),
):
    _required_flag(_name, _value)
for _name in ("B70_ONEDNN_PROFILE", "B70_ONEDNN_SHORT_CHUNK_ONLY",
              "B70_ONEDNN_FINAL_CHUNK_ONLY", "B70_ONEDNN_VALIDATE",
              "B70_ONEDNN_VALIDATE_FP32", "B70_ONEDNN_MIXED_TRACE",
              "B70_ONEDNN_MIXED_VALIDATE"):
    if os.environ.get(_name) not in (None, "0"):
        raise RuntimeError(f"{_name} is not part of the production policy")
torch.ops.load_library("/opt/b70/native_sdpa.so")


def _count(name, query_tokens, kv_tokens=0):
    """Power-of-two reports keep telemetry bounded as serving runs indefinitely."""
    row = _COUNTERS[name]
    row["calls"] += 1
    row["query_tokens"] += query_tokens
    row["kv_tokens"] += kv_tokens
    if row["calls"] & (row["calls"] - 1) == 0:
        print("B70_ATTENTION_ROUTE", name, dict(row), flush=True)


def eligible(d):
    if not q128_eligible(d) or torch.xpu.is_current_stream_capturing():
        return False
    q, k, v = d["q"], d["k"], d["v"]
    length = d["max_seqlen_k"]
    return (
        MIN_KV <= length <= MAX_KV and q.shape[0] <= length and
        d["block_table"].shape[1] >= (length + PAGE - 1) // PAGE and
        k.dtype == v.dtype == torch.float8_e4m3fn and
        k.shape[1:] == v.shape[1:] == (PAGE, 4, 256)
    )


def _onednn(d):
    q, k, v = d["q"], d["k"], d["v"]
    length = d["max_seqlen_k"]
    rows = q.shape[0]
    pages = d["block_table"][0, :(length + PAGE - 1) // PAGE].contiguous()
    k_scale = d["k_descale"].as_strided((1,), (1,))
    v_scale = d["v_descale"].as_strided((1,), (1,))
    divisor = torch.full((1,), 16.0, device=q.device, dtype=torch.float16)
    negative_inf = torch.full((1,), float("-inf"), device=q.device,
                              dtype=torch.float32)
    out = d.get("out")
    if out is None:
        out = torch.empty_like(q)
    key = torch.empty((1, length, 256), device=q.device, dtype=torch.float16)
    value = torch.empty_like(key)
    query = torch.empty((6, rows, 256), device=q.device, dtype=torch.float16)
    result = torch.empty_like(query)
    for head in range(4):
        torch.ops.b70_sdpa_probe.gather_dequant(k, pages, k_scale, key[0], head)
        torch.ops.b70_sdpa_probe.gather_dequant(v, pages, v_scale, value[0], head)
        query.copy_(q[:, head * 6:(head + 1) * 6].permute(1, 0, 2))
        torch.ops.b70_sdpa_probe.forward(query, key, value, result, divisor,
                                         negative_inf)
        out[:, head * 6:(head + 1) * 6].copy_(result.permute(1, 0, 2))
    return out


def _subcall(d, starts, lengths, index, out):
    first, last = starts[index:index + 2]
    sub = dict(d)
    sub["q"] = d["q"][first:last]
    sub["out"] = out
    sub["cu_seqlens_q"] = torch.tensor([0, last - first],
                                        device=d["cu_seqlens_q"].device,
                                        dtype=d["cu_seqlens_q"].dtype)
    sub["seqused_k"] = d["seqused_k"][index:index + 1].contiguous()
    sub["block_table"] = d["block_table"][index:index + 1].contiguous()
    sub["max_seqlen_q"] = last - first
    sub["max_seqlen_k"] = lengths[index]
    for scale in ("k_descale", "v_descale"):
        if d.get(scale) is not None and d[scale].ndim > 1:
            sub[scale] = d[scale][index:index + 1]
    return sub


def flash_attn_varlen_func(**d):
    q = d["q"]
    if torch.xpu.is_current_stream_capturing():
        _count("capture", q.shape[0])
        return fallback(**d)
    if eligible(d):
        _count("onednn_single", q.shape[0], d["max_seqlen_k"])
        return _onednn(d)

    if QUALIFICATION_ROUTE_OFF:
        _count("base_mixed", q.shape[0])
        return fallback(**d)

    cu, used, table = (d.get(name) for name in
                       ("cu_seqlens_q", "seqused_k", "block_table"))
    if (q.shape[0] < 256 or d.get("max_seqlen_q", 0) < 256 or
            d.get("max_seqlen_k", 0) < MIN_KV or
            cu is None or used is None or table is None or
            cu.numel() <= 2 or table.shape[0] != used.numel() or
            cu.numel() != used.numel() + 1 or
            d.get("scheduler_metadata") is not None):
        _count("base_single", q.shape[0])
        return fallback(**d)

    # These transfers synchronize once per attention layer. Qualification
    # measures their end-to-end cost; lengths must remain exact for each row.
    starts, lengths = cu.tolist(), used.tolist()
    if (starts[0] != 0 or starts[-1] != q.shape[0] or
            any(a > b for a, b in zip(starts, starts[1:])) or
            any(starts[i + 1] - starts[i] > length or length < 0
                for i, length in enumerate(lengths))):
        _count("unsupported_mixed", q.shape[0])
        return fallback(**d)
    subs = [_subcall(d, starts, lengths, i, None)
            for i in range(len(lengths)) if starts[i] != starts[i + 1]]
    if not any(eligible(sub) for sub in subs):
        _count("base_mixed", q.shape[0])
        return fallback(**d)
    out = d.get("out")
    if out is None:
        out = torch.empty_like(q)
    for i, (first, last) in enumerate(zip(starts, starts[1:])):
        if first == last:
            _count("empty_mixed", 0)
            continue
        sub = _subcall(d, starts, lengths, i, out[first:last])
        if eligible(sub):
            _count("onednn_mixed", last - first, lengths[i])
            _onednn(sub)
        else:
            _count("base_mixed", last - first, lengths[i])
            fallback(**sub)
    return out

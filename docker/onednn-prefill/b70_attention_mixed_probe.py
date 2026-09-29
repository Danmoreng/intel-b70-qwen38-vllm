"""Experimental long-prefill route; leave Q128/M04/native as fallback.

This adapter is for qualification runs only. The native operator and route
still require serving and quality qualification.
"""

import os

import torch

from b70_attention_base import flash_attn_varlen_func as fallback
from b70_attention_base import q128_eligible


ENABLED = os.environ.get("B70_ONEDNN_PREFILL") == "1"
VALIDATE = os.environ.get("B70_ONEDNN_VALIDATE") == "1"
VALIDATE_FP32 = os.environ.get("B70_ONEDNN_VALIDATE_FP32") == "1"
MIXED_TRACE = os.environ.get("B70_ONEDNN_MIXED_TRACE") == "1"
MIXED_VALIDATE = os.environ.get("B70_ONEDNN_MIXED_VALIDATE") == "1"
MIXED_ROUTE = os.environ.get("B70_ONEDNN_MIXED_ROUTE") == "1"
if MIXED_VALIDATE and MIXED_ROUTE:
    raise ValueError("mixed validation and routing are separate experiments")
PROFILE = os.environ.get("B70_ONEDNN_PROFILE", "reference")
if PROFILE not in ("reference", "performance"):
    raise ValueError("B70_ONEDNN_PROFILE must be reference or performance")
# The old flag is retained only so frozen qualification runs stay reproducible.
# A short prefill chunk may also occur before the prompt's final chunk.
_short_policy = os.environ.get("B70_ONEDNN_SHORT_CHUNK_ONLY",
                               "0" if PROFILE == "performance" else "1")
if _short_policy not in ("0", "1"):
    raise ValueError("B70_ONEDNN_SHORT_CHUNK_ONLY must be 0 or 1")
SHORT_CHUNK_ONLY = (_short_policy == "1" or
                    os.environ.get("B70_ONEDNN_FINAL_CHUNK_ONLY") == "1")
MIN_KV = int(os.environ.get("B70_ONEDNN_MIN_KV", "16384"))
if MIN_KV < 1:
    raise ValueError("B70_ONEDNN_MIN_KV must be positive")
MAX_KV = int(os.environ.get("B70_ONEDNN_MAX_KV",
                            "200704" if PROFILE == "performance" else "131072"))
if MAX_KV < MIN_KV:
    raise ValueError("B70_ONEDNN_MAX_KV must be at least B70_ONEDNN_MIN_KV")
_seen = set()
_seen_mixed = set()
_validated_mixed = set()
_validated_mixed_decode = set()
_validated = {}
_validated_fp32 = {}
if ENABLED:
    torch.ops.load_library("/opt/b70/native_sdpa.so")


def eligible(d):
    if not ENABLED or not q128_eligible(d):
        return False
    if torch.xpu.is_current_stream_capturing():
        return False
    q = d["q"]
    k, v = d["k"], d["v"]
    length = d["max_seqlen_k"]
    if SHORT_CHUNK_ONLY and q.shape[0] >= 6656:
        return False
    return (
        q.shape[0] >= 256 and MIN_KV <= length <= MAX_KV and q.shape[0] <= length
        and d["block_table"].shape[1] >= (length + 1663) // 1664
        and d["block_table"].dtype == torch.int32
        and k.dtype == v.dtype == torch.float8_e4m3fn
        and k.stride(3) == v.stride(3) == 1
        and k.shape[1:] == v.shape[1:] == (1664, 4, 256)
    )


def mixed_subcall(d, starts, used, index, out):
    q_rows = starts[index + 1] - starts[index]
    sub = dict(d)
    sub["q"] = d["q"][starts[index]:starts[index + 1]]
    sub["out"] = out
    sub["cu_seqlens_q"] = torch.tensor(
        [0, q_rows], device=d["cu_seqlens_q"].device,
        dtype=d["cu_seqlens_q"].dtype)
    sub["seqused_k"] = d["seqused_k"][index:index + 1].contiguous()
    sub["block_table"] = d["block_table"][index:index + 1].contiguous()
    sub["max_seqlen_q"] = q_rows
    sub["max_seqlen_k"] = used[index]
    for scale in ("k_descale", "v_descale"):
        if d.get(scale) is not None and d[scale].ndim > 1:
            sub[scale] = d[scale][index:index + 1]
    return sub


def flash_attn_varlen_func(**d):
    if not eligible(d):
        if MIXED_TRACE and ENABLED and not torch.xpu.is_current_stream_capturing():
            cu = d.get("cu_seqlens_q")
            used = d.get("seqused_k")
            if (cu is not None and used is not None and cu.numel() > 2
                    and d["q"].shape[0] >= 256
                    and d.get("max_seqlen_k", 0) >= 1024):
                signature = (d["q"].shape[0], d.get("max_seqlen_k"), cu.numel())
                if signature not in _seen_mixed and len(_seen_mixed) < 16:
                    block_table = d.get("block_table")
                    print("B70_ONEDNN_MIXED_TRACE", {
                        "signature": signature,
                        "cu_seqlens_q": cu.tolist(),
                        "seqused_k": used.tolist(),
                        "max_seqlen_q": d.get("max_seqlen_q"),
                        "block_table_shape": (tuple(block_table.shape)
                                              if block_table is not None else None),
                        "block_table_stride": (tuple(block_table.stride())
                                               if block_table is not None else None),
                        "q_stride": tuple(d["q"].stride()),
                        "out_shape": (tuple(d["out"].shape)
                                      if d.get("out") is not None else None),
                        "scheduler_metadata": d.get("scheduler_metadata") is not None,
                    }, flush=True)
                    _seen_mixed.add(signature)
        if (MIXED_ROUTE and ENABLED and
                not torch.xpu.is_current_stream_capturing() and
                d["q"].shape[0] >= 256):
            cu = d.get("cu_seqlens_q")
            used = d.get("seqused_k")
            table = d.get("block_table")
            if (cu is not None and used is not None and table is not None and
                    cu.numel() > 2 and table.shape[0] == used.numel() and
                    d.get("scheduler_metadata") is None):
                starts = cu.tolist()
                lengths = used.tolist()
                if (starts[0] == 0 and starts[-1] == d["q"].shape[0] and
                        all(left <= right for left, right in zip(starts, starts[1:])) and
                        all(starts[i + 1] - starts[i] <= length
                            for i, length in enumerate(lengths))):
                    subs = [mixed_subcall(d, starts, lengths, i, None)
                            for i in range(len(lengths))]
                    long_indices = {i for i, sub in enumerate(subs) if eligible(sub)}
                    if long_indices:
                        output = d.get("out")
                        if output is None:
                            output = torch.empty_like(d["q"])
                        for i, sub in enumerate(subs):
                            sub["out"] = output[starts[i]:starts[i + 1]]
                            if i in long_indices:
                                flash_attn_varlen_func(**sub)
                            else:
                                fallback(**sub)
                        return output
        reference = fallback(**d)
        if (MIXED_VALIDATE and ENABLED and
                not torch.xpu.is_current_stream_capturing()):
            cu = d.get("cu_seqlens_q")
            used = d.get("seqused_k")
            table = d.get("block_table")
            if (cu is not None and used is not None and table is not None and
                    cu.numel() > 2 and table.shape[0] == used.numel() and
                    d["q"].shape[0] >= 256 and
                    (len(_validated_mixed) < 8 or
                     len(_validated_mixed_decode) < 8)):
                starts = cu.tolist()
                lengths = used.tolist()
                for index, length in enumerate(lengths):
                    q_rows = starts[index + 1] - starts[index]
                    signature = (index, q_rows, length)
                    long_prefill = (q_rows >= 256 and
                                    MIN_KV <= length <= MAX_KV and
                                    signature not in _validated_mixed and
                                    len(_validated_mixed) < 8)
                    short_decode = (1 <= q_rows <= 5 and
                                    signature not in _validated_mixed_decode and
                                    len(_validated_mixed_decode) < 8)
                    if not long_prefill and not short_decode:
                        continue
                    sub = mixed_subcall(d, starts, lengths, index, None)
                    if long_prefill:
                        if not eligible(sub):
                            raise RuntimeError(f"mixed oneDNN subcall rejected: {signature}")
                        candidate = flash_attn_varlen_func(**sub)
                    else:
                        candidate = fallback(**sub)
                    control = reference[starts[index]:starts[index + 1]]
                    torch.xpu.synchronize()
                    delta = (candidate.float() - control.float()).abs()
                    report = {
                        "signature": signature,
                        "allclose": bool(torch.allclose(
                            candidate, control, rtol=.01, atol=.002)),
                        "relative_l2": float((delta.norm() /
                                              control.float().norm()).item()),
                        "max_abs": float(delta.max().item()),
                    }
                    marker = ("B70_ONEDNN_MIXED_VALIDATE" if long_prefill else
                              "B70_ONEDNN_MIXED_DECODE_VALIDATE")
                    print(marker, report, flush=True)
                    validated = (_validated_mixed if long_prefill else
                                 _validated_mixed_decode)
                    validated.add(signature)
                    if not report["allclose"]:
                        raise RuntimeError("mixed attention subcall failed validation")
        return reference
    q, k, v = d["q"], d["k"], d["v"]
    length = d["max_seqlen_k"]
    q_rows = q.shape[0]
    pages = d["block_table"][0, :(length + 1663) // 1664].contiguous()
    k_scale = d["k_descale"].as_strided((1,), (1,))
    v_scale = d["v_descale"].as_strided((1,), (1,))
    divisor = torch.full((1,), 16.0, device=q.device, dtype=torch.float16)
    negative_inf = torch.full((1,), float("-inf"), device=q.device, dtype=torch.float32)
    out = d.get("out")
    if out is None:
        out = torch.empty_like(q)
    key = torch.empty((1, length, 256), device=q.device, dtype=torch.float16)
    value = torch.empty_like(key)
    query = torch.empty((6, q_rows, 256), device=q.device, dtype=torch.float16)
    result = torch.empty_like(query)
    for head in range(4):
        torch.ops.b70_sdpa_probe.gather_dequant(k, pages, k_scale, key[0], head)
        torch.ops.b70_sdpa_probe.gather_dequant(v, pages, v_scale, value[0], head)
        query.copy_(q[:, head * 6:(head + 1) * 6].permute(1, 0, 2))
        torch.ops.b70_sdpa_probe.forward(query, key, value, result, divisor, negative_inf)
        out[:, head * 6:(head + 1) * 6].copy_(result.permute(1, 0, 2))
    signature = (q_rows, length)
    check_fp32 = VALIDATE_FP32 and _validated_fp32.get(signature, 0) < 2
    check_regular = VALIDATE and _validated.get(signature, 0) < 2
    if check_regular or check_fp32:
        reference = fallback(**{**d, "out": None})
        torch.xpu.synchronize()
        difference = (out.float() - reference.float()).abs()
        if check_regular:
            row_max = difference.amax(dim=(1, 2))
            worst = torch.topk(row_max, min(4, q_rows))
            print("B70_ONEDNN_VALIDATE", {
                "signature": signature,
                "used_k": int(d["seqused_k"].item()),
                "k_scale": float(k_scale.item()),
                "v_scale": float(v_scale.item()),
                "max_abs": float(difference.max().item()),
                "mean_abs": float(difference.mean().item()),
                "relative_l2": float((difference.norm() / reference.float().norm()).item()),
                "allclose": bool(torch.allclose(out, reference, rtol=.01, atol=.002)),
                "worst_rows": worst.indices.tolist(),
                "worst_row_abs": worst.values.tolist(),
            }, flush=True)
            _validated[signature] = _validated.get(signature, 0) + 1
        if check_fp32:
            # Diagnostic only: recompute selected real rows over their exact
            # visible FP8->FP16 KV prefix with FP32 QK, softmax, and PV.
            head_difference = difference[:, :6].amax(dim=(1, 2))
            worst_rows = torch.topk(head_difference, min(4, q_rows)).indices.tolist()
            rows_host = sorted(set([0, q_rows // 2, q_rows - 1, *worst_rows]))
            rows = torch.tensor(rows_host, device=q.device, dtype=torch.long)
            torch.ops.b70_sdpa_probe.gather_dequant(k, pages, k_scale, key[0], 0)
            torch.ops.b70_sdpa_probe.gather_dequant(v, pages, v_scale, value[0], 0)
            selected_q = q.index_select(0, rows)[:, :6].permute(1, 0, 2).float()
            scores = torch.matmul(selected_q, key[0].float().T) * .0625
            columns = torch.arange(length, device=q.device)
            last_visible = length - q_rows + rows
            scores.masked_fill_(columns[None, None, :] > last_visible[None, :, None],
                                float("-inf"))
            selected_fp32 = torch.matmul(torch.softmax(scores, dim=-1),
                                         value[0].float()).permute(1, 0, 2)
            selected_onednn = out.index_select(0, rows)[:, :6].float()
            selected_q128 = reference.index_select(0, rows)[:, :6].float()
            denom = selected_fp32.norm()
            print("B70_ONEDNN_FP32_DIAGNOSTIC", {
                "signature": signature, "head": 0, "rows": rows_host,
                "k_scale": float(k_scale.item()), "v_scale": float(v_scale.item()),
                "onednn_relative_l2": float(((selected_onednn - selected_fp32).norm() / denom).item()),
                "q128_relative_l2": float(((selected_q128 - selected_fp32).norm() / denom).item()),
                "onednn_max_abs": float((selected_onednn - selected_fp32).abs().max().item()),
                "q128_max_abs": float((selected_q128 - selected_fp32).abs().max().item()),
            }, flush=True)
            _validated_fp32[signature] = _validated_fp32.get(signature, 0) + 1
    if signature not in _seen:
        print("B70_ONEDNN_PREFILL_DISPATCH", signature, flush=True)
        _seen.add(signature)
    return out

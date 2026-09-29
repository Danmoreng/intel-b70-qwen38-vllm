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


def flash_attn_varlen_func(**d):
    if not eligible(d):
        return fallback(**d)
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

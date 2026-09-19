"""Q128 production attention plus the M04 shared-KV verification arm."""
import json
import os
from pathlib import Path
import threading
import time

import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func as original


torch.ops.load_library("/opt/b70/tiles.so")
torch.ops.load_library("/opt/b70/m04.so")
_validate = False
_seen = set()
_logged = set()
_root = Path("/evidence")


def control():
    global _validate
    while not (_root / "validation-done").exists():
        _validate = (_root / "validate").exists()
        time.sleep(0.1)
    _validate = False


threading.Thread(target=control, daemon=True).start()


def common(d):
    q, k, v = d["q"], d["k"], d["v"]
    cq = d.get("cu_seqlens_q")
    used = d.get("seqused_k")
    bt = d.get("block_table")
    ks, vs = d.get("k_descale"), d.get("v_descale")
    return (
        q.is_xpu and q.dtype == torch.float16 and q.ndim == 3 and
        q.shape[1:] == (24, 256) and q.is_contiguous() and
        k.dtype == torch.float8_e4m3fn and v.dtype == k.dtype and
        k.ndim == 4 and k.shape[1:] == (1664, 4, 256) and
        v.shape == k.shape and
        k.stride() == v.stride() == (3407872, 2048, 512, 1) and
        cq is not None and cq.numel() == 2 and cq.dtype == torch.int32 and
        cq.is_contiguous() and used is not None and used.numel() == 1 and
        used.dtype == torch.int32 and used.is_contiguous() and
        bt is not None and bt.ndim == 2 and bt.shape[0] == 1 and
        bt.dtype == torch.int32 and bt.is_contiguous() and
        d.get("max_seqlen_q") == q.shape[0] and
        0 < d.get("max_seqlen_k", 0) and d.get("causal") is True and
        d.get("softmax_scale") == 0.0625 and
        tuple(d.get("window_size") or (-1, -1)) == (-1, -1) and
        not any(d.get(n) for n in ("return_softmax_lse", "return_attn_probs",
                                   "dropout_p", "softcap")) and
        all(d.get(n) is None for n in
            ("s_aux", "alibi_slopes", "q_descale", "cu_seqlens_k", "q_v",
             "scheduler_metadata")) and
        all(s is not None and s.dtype == torch.float32 and s.is_xpu and
            (s.numel() == 1 or all(t == 0 for t in s.stride()))
            for s in (ks, vs)))


def q128_eligible(d):
    q = d["q"]
    out = d.get("out")
    return (common(d) and 256 <= q.shape[0] <= 6656 and
            (out is None or (out.shape == q.shape and out.dtype == q.dtype and
                             out.is_contiguous())))


def m04_eligible(d):
    q = d["q"]
    out = d.get("out")
    return (common(d) and 2 <= q.shape[0] <= 5 and
            (out is None or (out.shape == q.shape and out.dtype == q.dtype and
                             out.is_contiguous())))


def run_m04(d):
    q = d["q"]
    q_len = q.shape[0]
    packed_heads = q_len * 24
    packed_q = (q.view(q_len, 4, 6, 256).permute(1, 0, 2, 3)
                .reshape(1, packed_heads, 256).contiguous())
    packed_out = torch.empty_like(packed_q)
    num_splits = {2: 32, 3: 8, 4: 16, 5: 16}[q_len]
    temp = torch.empty((1, packed_heads * num_splits, 256), device=q.device,
                       dtype=q.dtype)
    sums = torch.empty((1, packed_heads, num_splits), device=q.device,
                       dtype=torch.float32)
    maxima = torch.empty_like(sums)
    ks = d["k_descale"].as_strided((1,), (1,))
    vs = d["v_descale"].as_strided((1,), (1,))
    torch.ops.b70_ops.shared_kv_verify_out(
        packed_q, d["k"], d["v"], d["block_table"], d["cu_seqlens_q"],
        d["seqused_k"], ks, vs, packed_out, temp, sums, maxima,
        d["max_seqlen_k"], num_splits, 8)
    result = (packed_out.view(4, q_len, 6, 256).permute(1, 0, 2, 3)
              .reshape(q_len, 24, 256).contiguous())
    out = d.get("out")
    if out is not None:
        out.copy_(result)
        result = out
    key = (q_len, d["max_seqlen_k"], num_splits)
    if key not in _logged:
        print("B70_M04_SHARED_KV_DISPATCH", key, flush=True)
        _logged.add(key)
    return result


def flash_attn_varlen_func(**d):
    if m04_eligible(d):
        return run_m04(d)
    if not q128_eligible(d):
        return original(**d)
    q, k, v = d["q"], d["k"], d["v"]
    ks = d["k_descale"].as_strided((1,), (1,))
    vs = d["v_descale"].as_strided((1,), (1,))
    y = torch.ops.b70_tiles.forward(
        q, k, v, d["block_table"], d["cu_seqlens_q"], d["seqused_k"],
        ks, vs, d["max_seqlen_k"], 1)
    key = (q.shape[0], d["max_seqlen_k"])
    if _validate and key not in _seen:
        ref = original(**{**d, "out": None})
        torch.xpu.synchronize()
        delta = (y.float() - ref.float()).abs()
        ok = torch.allclose(y, ref, rtol=0.01, atol=0.002)
        row = {"q": key[0], "kv": key[1], "max_abs": delta.max().item(),
               "allclose": ok, "finite": bool(y.isfinite().all().item()),
               "kv_stride": list(k.stride()), "variant": "Q128_KV32"}
        with (_root / "real-activation-checks.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        assert ok and row["finite"], row
        _seen.add(key)
    if key not in _logged:
        print("B70_Q128_DISPATCH", key, "kv_stride", k.stride(), flush=True)
        _logged.add(key)
    out = d.get("out")
    if out is not None:
        out.copy_(y)
        return out
    return y

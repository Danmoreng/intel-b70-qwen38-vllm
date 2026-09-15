"""Q128 prefill adapter; writes directly into vLLM's output when supplied."""
from __future__ import annotations
import os
import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func as fallback

def load_library(path: str | None = None) -> None:
    torch.ops.load_library(path or os.environ.get("B70_OPS_LIBRARY", "/opt/b70_ops/b70_ops.so"))

def _eligible(d: dict) -> bool:
    q,k,v=d["q"],d["k"],d["v"]
    cq,used,bt=d.get("cu_seqlens_q"),d.get("seqused_k"),d.get("block_table")
    ks,vs,out=d.get("k_descale"),d.get("v_descale"),d.get("out")
    return (q.is_xpu and q.dtype==torch.float16 and q.ndim==3 and 256<=q.shape[0]<=6656
      and q.shape[1:]==(24,256) and q.is_contiguous() and k.is_xpu and v.is_xpu
      and k.dtype==torch.float8_e4m3fn and v.dtype==k.dtype and k.ndim==4
      and k.shape[1:]==(1664,4,256) and v.shape==k.shape
      and k.stride()==v.stride()==(3407872,2048,512,1)
      and cq is not None and cq.is_xpu and cq.dtype==torch.int32 and cq.numel()==2 and cq.is_contiguous()
      and used is not None and used.is_xpu and used.dtype==torch.int32 and used.numel()==1 and used.is_contiguous()
      and bt is not None and bt.is_xpu and bt.dtype==torch.int32 and bt.ndim==2 and bt.shape[0]==1 and bt.is_contiguous()
      and d.get("max_seqlen_q")==q.shape[0] and 0<d.get("max_seqlen_k",0)
      and d.get("causal") is True and d.get("softmax_scale")==0.0625
      and tuple(d.get("window_size") or (-1,-1))==(-1,-1)
      and not any(d.get(n) for n in ("return_softmax_lse","return_attn_probs","dropout_p","softcap"))
      and all(d.get(n) is None for n in ("s_aux","alibi_slopes","q_descale","cu_seqlens_k","q_v","scheduler_metadata"))
      and all(s is not None and s.is_xpu and s.dtype==torch.float32 and s.numel()==1 for s in (ks,vs))
      and (out is None or (out.is_xpu and out.shape==q.shape and out.dtype==q.dtype
                           and out.stride()==q.stride() and out.is_contiguous()
                           and out.data_ptr()!=q.data_ptr())))

def flash_attn_varlen_func(**d):
    if not _eligible(d): return fallback(**d)
    common=(d["q"],d["k"],d["v"],d["block_table"],d["cu_seqlens_q"],d["seqused_k"],
            d["k_descale"].as_strided((1,),(1,)),d["v_descale"].as_strided((1,),(1,)))
    if d.get("out") is None:
        return torch.ops.b70_ops.q128_forward(*common,d["max_seqlen_k"],1)
    return torch.ops.b70_ops.q128_forward_out(*common,d["out"],d["max_seqlen_k"],1)

load_library()

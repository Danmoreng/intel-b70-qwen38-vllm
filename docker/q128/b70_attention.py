import os,json,time,threading
from pathlib import Path
import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func as original

torch.ops.load_library('/opt/b70/tiles.so')
_validate=False
_seen=set()
_logged=set()
_root=Path('/evidence')
def control():
 global _validate
 while not (_root/'validation-done').exists():
  _validate=(_root/'validate').exists();time.sleep(.1)
 _validate=False
threading.Thread(target=control,daemon=True).start()

def eligible(d):
 q,k,v=d['q'],d['k'],d['v'];cq=d.get('cu_seqlens_q');used=d.get('seqused_k');bt=d.get('block_table');ks=d.get('k_descale');vs=d.get('v_descale');out=d.get('out')
 return (q.is_xpu and q.dtype==torch.float16 and q.ndim==3 and 256<=q.shape[0]<=6656 and q.shape[1:]==(24,256) and q.is_contiguous()
  and k.dtype==torch.float8_e4m3fn and v.dtype==k.dtype and k.ndim==4 and k.shape[1:]==(1664,4,256) and v.shape==k.shape and k.stride()==v.stride()==(3407872,2048,512,1)
  and cq is not None and cq.numel()==2 and cq.dtype==torch.int32 and cq.is_contiguous() and used is not None and used.numel()==1 and used.dtype==torch.int32 and used.is_contiguous()
  and bt is not None and bt.ndim==2 and bt.shape[0]==1 and bt.dtype==torch.int32 and bt.is_contiguous()
  and d.get('max_seqlen_q')==q.shape[0] and 0<d.get('max_seqlen_k',0)
  and d.get('causal') is True and d.get('softmax_scale')==.0625 and tuple(d.get('window_size') or (-1,-1))==(-1,-1)
  and not any(d.get(n) for n in ['return_softmax_lse','return_attn_probs','dropout_p','softcap'])
  and all(d.get(n) is None for n in ['s_aux','alibi_slopes','q_descale','cu_seqlens_k','q_v','scheduler_metadata'])
  and all(s is not None and s.dtype==torch.float32 and s.is_xpu and (s.numel()==1 or all(t==0 for t in s.stride())) for s in [ks,vs])
  and (out is None or (out.shape==q.shape and out.dtype==q.dtype and out.is_contiguous())))

def flash_attn_varlen_func(**d):
 if not eligible(d):return original(**d)
 q,k,v=d['q'],d['k'],d['v'];ks=d['k_descale'].as_strided((1,),(1,));vs=d['v_descale'].as_strided((1,),(1,))
 y=torch.ops.b70_tiles.forward(q,k,v,d['block_table'],d['cu_seqlens_q'],d['seqused_k'],ks,vs,d['max_seqlen_k'],1)
 key=(q.shape[0],d['max_seqlen_k'])
 if _validate and key not in _seen:
  ref=original(**{**d,'out':None});torch.xpu.synchronize()
  delta=(y.float()-ref.float()).abs();ok=torch.allclose(y,ref,rtol=.01,atol=.002)
  row={'q':key[0],'kv':key[1],'max_abs':delta.max().item(),'allclose':ok,'finite':bool(y.isfinite().all().item()),'kv_stride':list(k.stride()),'variant':'Q128_KV32'}
  with (_root/'real-activation-checks.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  assert ok and row['finite'],row
  _seen.add(key)
 if key not in _logged:
  print('B70_Q128_DISPATCH',key,'kv_stride',k.stride(),flush=True);_logged.add(key)
 out=d.get('out')
 if out is not None:out.copy_(y);return out
 return y

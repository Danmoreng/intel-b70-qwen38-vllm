"""CPU-only installer scoping/fallback/source/library rejection checks."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
import verify_attention as V

library=Path('/library/m04.so')
os.environ['EXL3_M04_LIBRARY']=str(library)
os.environ['EXL3_M04_LIBRARY_SHA256']=hashlib.sha256(library.read_bytes()).hexdigest()
origin=importlib.util.find_spec('vllm').origin
vllm=ModuleType('vllm');vllm.__file__=origin;vllm.__version__='0.30.0'
fa=ModuleType('vllm.v1.attention.backends.flash_attn')
fa.FlashAttentionBackend=SimpleNamespace(forward_includes_kv_cache_update=False)
native_marker, optimized_marker=object(),object()
fa.flash_attn_varlen_func=lambda *args,**kw:native_marker
class Impl:
    attn_type='decoder';dcp_world_size=1;head_size=256;num_heads=24;num_kv_heads=4;kv_cache_dtype='fp8'
    def forward(self,**kw):return fa.flash_attn_varlen_func(**kw)
fa.FlashAttentionImpl=Impl
for name in ['vllm','vllm.v1','vllm.v1.attention','vllm.v1.attention.backends','vllm.v1.attention.backend']:
    sys.modules[name]=vllm if name=='vllm' else ModuleType(name)
sys.modules['vllm.v1.attention.backends.flash_attn']=fa
sys.modules['vllm.v1.attention.backend'].AttentionType=SimpleNamespace(DECODER='decoder')
torch.xpu.is_current_stream_capturing=lambda:False
V.eligible=lambda kw:kw.get('supported',True)
V.run=lambda kw:optimized_marker
os.environ['EXL3_M04_LIBRARY_SHA256']='0'*64
try:import install_serving
except AssertionError:pass
else:raise AssertionError('Wrong library digest accepted')
assert fa.flash_attn_varlen_func() is native_marker
os.environ['EXL3_M04_LIBRARY_SHA256']=hashlib.sha256(library.read_bytes()).hexdigest()
import install_serving
d={'cu_seqlens_q':torch.arange(5),'max_seqlen_q':4,'k':SimpleNamespace(shape=(4,1600,4,256)),'max_seqlen_k':4096}
assert fa.FlashAttentionImpl().forward(**d) is optimized_marker
assert fa.flash_attn_varlen_func(**d) is native_marker,'Context leaked outside decoder call'
for field,value in [('attn_type','encoder'),('attn_type','cross'),('dcp_world_size',2),('head_size',128),('kv_cache_dtype','float16')]:
    obj=fa.FlashAttentionImpl();setattr(obj,field,value)
    assert obj.forward(**d) is native_marker,(field,value)
assert fa.FlashAttentionImpl().forward(**{**d,'cu_seqlens_q':torch.arange(6)}) is native_marker
assert fa.FlashAttentionImpl().forward(**{**d,'supported':False}) is native_marker
try:install_serving.install()
except AssertionError:pass
else:raise AssertionError('Duplicate install accepted')
old=install_serving.SOURCE_HASHES.copy();install_serving.SOURCE_HASHES={k:'0'*64 for k in old}
try:install_serving.install()
except AssertionError:pass
else:raise AssertionError('Unknown source accepted')
print('PASS installer decoder/thread scoping, native fallback, wrong library/source and duplicate rejection')

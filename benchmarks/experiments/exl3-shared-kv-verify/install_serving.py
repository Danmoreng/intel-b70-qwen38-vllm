"""Opt-in source-guarded worker installation of the separately gated prototype."""
from contextvars import ContextVar
import functools
import hashlib
import os
from pathlib import Path

import torch

SOURCE_HASHES={
    'v1/attention/backends/flash_attn.py':'b170a4789e18104c1bee35bea70df51abe6766693d8f4726a59f2e6f5c4282aa',
    'v1/attention/backends/fa_utils.py':'08fa6e231d3af3a71a6e59162dd12328aa8beb28548165ae58f130a92adb5a83',
}


class M04Extension:
    def m04_installed(self):
        from vllm.v1.attention.backends import flash_attn
        return getattr(flash_attn,'_b70_exl3_m04_installed',False)


def install():
    library=os.environ.get('EXL3_M04_LIBRARY')
    if not library:return
    import vllm
    assert vllm.__version__=='0.30.0' and torch.__version__=='2.13.0+xpu'
    root=Path(vllm.__file__).parent
    actual={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_HASHES}
    assert actual==SOURCE_HASHES, f'Unknown attention runtime: {actual}'
    assert hashlib.sha256(Path(library).read_bytes()).hexdigest()==os.environ['EXL3_M04_LIBRARY_SHA256']
    from vllm.v1.attention.backends import flash_attn as fa
    from vllm.v1.attention.backend import AttentionType
    # The V2 attention layer owns cache writes separately. This hook intercepts
    # only the read/compute function and never calls an update operation.
    assert fa.FlashAttentionBackend.forward_includes_kv_cache_update is False
    assert not getattr(fa,'_b70_exl3_m04_installed',False)
    torch.ops.load_library(library)
    assert torch._C._dispatch_has_kernel_for_dispatch_key('b70_exl3_attention::shared_kv_verify_out','XPU')
    import verify_attention as V
    original=fa.flash_attn_varlen_func
    seen=set()
    decoder_context=ContextVar('exl3_m04_decoder_context',default=False)

    def forward(*args,**d):
        if not decoder_context.get() or args or not V.eligible(d):return original(*args,**d)
        # Supported batch1..4 only during this first serving qualification.
        # Other batches use native unchanged; admission remains C16.
        batch=d['cu_seqlens_q'].numel()-1
        if batch>4:return original(**d)
        capture=torch.xpu.is_current_stream_capturing()
        key=(batch,d['max_seqlen_q'],d['k'].shape[1],capture)
        if key not in seen:
            print('EXL3_M04_DISPATCH',key,'max_kv',d['max_seqlen_k'],flush=True);seen.add(key)
        return V.run(d)

    # Gate unsupported encoder/cross-attention semantics before the low-level
    # hook; the global FA function itself does not expose the attention type.
    original_forward=fa.FlashAttentionImpl.forward
    @functools.wraps(original_forward)
    def layer_forward(self,*args,**kwargs):
        allowed=(self.attn_type==AttentionType.DECODER and self.dcp_world_size==1 and
                 self.head_size==256 and self.num_heads==24 and self.num_kv_heads==4 and
                 self.kv_cache_dtype in ('fp8','fp8_e4m3'))
        token=decoder_context.set(allowed)
        try:return original_forward(self,*args,**kwargs)
        finally:decoder_context.reset(token)
    # Scope the hook per call/thread; unsupported layer types continue through
    # the unchanged native method and native compute function.
    fa.FlashAttentionImpl.forward=layer_forward
    fa.flash_attn_varlen_func=forward
    fa._b70_exl3_m04_installed=True
    print('EXL3_M04_INSTALL_PASS',actual,'library',os.environ['EXL3_M04_LIBRARY_SHA256'],flush=True)


install()

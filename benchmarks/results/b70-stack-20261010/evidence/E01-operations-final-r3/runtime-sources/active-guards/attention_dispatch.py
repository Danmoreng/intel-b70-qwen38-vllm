"""Opt-in, source-guarded attention compute; native V2 owns every KV write."""
from contextvars import ContextVar
from dataclasses import dataclass
import functools
import hashlib
import math
import os
from pathlib import Path

import torch

from . import attention_metadata as metadata
from . import shared_kv_verify as verify

SOURCE_HASHES = {'v1/attention/backends/flash_attn.py': 'ea53fb41165323f13094604051c73d44a579994fe5a427243aa9faafc280b1c2',
 'v1/attention/backends/fa_utils.py': '3896d53947719ec9a226479fdbde60e83b1df116d634e2f4331499acf727dd99',
 'v1/attention/backend.py': '77ac35180c1e0f854c6307f5deb3c7c0ac4a9e8702578d406ce058981488a137',
 'v1/worker/gpu/model_runner.py': '725c47114e148ade3a26b4cc9c5f7295588fdd387afc3cb5fdb2d16513a1bb5c',
 'v1/worker/gpu/attn_utils.py': '0f9df8b0153bd0a2720ee957f6cd1600b9d865eb12b7bf432501ffaa12fb6a9c',
 'v1/worker/gpu/model_states/mamba_hybrid.py': '1ddd7dc921d74789e77d3319dc167820ad7fc47ed51d8d9a35ba3938f5c6a079',
 'model_executor/layers/attention/attention.py': 'bde77677fd0e95ea09c6a580dfaf09b57741e8183e00cb106e2d1cc64d8e03f2',
 'model_executor/layers/quantization/kv_cache.py': '32b45463d0b36c021cb5ae0937d60b5c797ca383a0ef3e291170337486feb1b0'}


@dataclass(frozen=True)
class Context:
    allowed: bool
    plan: object = None
    scales: object = None


def supported_features(d):
    """Unknown or unsupported semantics always go to native unchanged."""
    if set(d) - verify.KNOWN or d.get('causal') is not True:
        return False
    if type(d.get('softmax_scale')) not in (int, float) or d['softmax_scale'] != .0625:
        return False
    window = d.get('window_size')
    if window is not None and (type(window) not in (tuple, list) or
                              len(window) != 2 or any(type(x) is not int for x in window)):
        return False
    if tuple(window or (-1, -1)) != (-1, -1) or d.get('fa_version', 2) != 2:
        return False
    for name in ('return_softmax_lse', 'return_attn_probs', 'dropout_p', 'softcap', 'deterministic'):
        x = d.get(name)
        if type(x) not in (type(None), bool, int, float) or x not in (None, False, 0, 0.):
            return False
    for name in ('num_splits', 'num_splits_kv'):
        x = d.get(name)
        if type(x) not in (type(None), int) or x not in (None, 0, 1):
            return False
    return all(d.get(name) is None for name in
               ('s_aux', 'alibi_slopes', 'q_descale', 'cu_seqlens_k', 'q_v',
                'scheduler_metadata', 'host_kv_lens', 'dynamic_causal', 'mask_mod', 'aux_tensors'))


def prefill_eligible(d, context, capturing):
    plan = context.plan
    if capturing or plan is None or context.scales is None or not supported_features(d):
        return False
    names = ('q', 'k', 'v', 'out', 'cu_seqlens_q', 'seqused_k', 'block_table', 'k_descale', 'v_descale')
    if any(not isinstance(d.get(name), torch.Tensor) for name in names):
        return False
    q, k, v, out, cq, used, bt, ks, vs = (d[name] for name in names)
    if (q.device.type != 'xpu' or q.dtype != torch.float16 or
            q.shape != (plan.tokens, 24, 256) or not q.is_contiguous() or
            out.shape != q.shape or out.dtype != q.dtype or not out.is_contiguous() or
            k.dtype != torch.float8_e4m3fn or v.dtype != k.dtype or k.ndim != 4 or
            k.shape[1] <= 0 or k.shape[1] % 64 or tuple(k.shape[2:]) != (4, 256) or
            v.shape != k.shape or v.stride() != k.stride()):
        return False
    strides = k.stride()
    if (strides[3] != 1 or strides[2] < 256 or strides[1] < 4*strides[2] or
            strides[0] < k.shape[1]*strides[1]):
        return False
    if (cq.dtype != torch.int32 or cq.shape != (plan.requests+1,) or not cq.is_contiguous() or
            used.dtype != torch.int32 or used.shape != (plan.requests,) or not used.is_contiguous() or
            bt.dtype != torch.int32 or bt.ndim != 2 or bt.shape[0] != plan.requests or not bt.is_contiguous()):
        return False
    if any(s.maximum_kv > bt.shape[1]*k.shape[1] for s in plan.segments):
        return False
    for scale in (ks, vs):
        if (scale.dtype != torch.float32 or scale.numel() < 1 or
                scale.untyped_storage().nbytes() != scale.element_size() or
                (scale.numel() != 1 and any(stride != 0 for stride in scale.stride()))):
            return False
    return all(t.device == q.device for t in (k, v, out, cq, used, bt, ks, vs))


def select_route(d, context, *, capturing, native_op, m04, onednn):
    if not context.allowed:
        return 'native'
    if onednn and native_op is not None and prefill_eligible(d, context, capturing):
        return 'mixed/oneDNN'
    if m04 and verify.eligible(d) and d['cu_seqlens_q'].numel()-1 <= 4:
        return 'M04'
    return 'native'


def _dispatch_selected(route, original, d, context, *, capturing, native_op, m04):
    """Consume one checked route; subgroup calls each select their own route."""
    if route == 'mixed/oneDNN':
        from .fp8kv_prefill import prefill_attention_onednn
        for segment in context.plan.segments:
            if segment.prefill:
                prefill_attention_onednn(native_op, d['q'][segment.start:segment.stop],
                    d['k'], d['v'], d['block_table'][segment.first], segment.maximum_kv,
                    *context.scales, d['softmax_scale'], d['out'][segment.start:segment.stop])
            else:
                group = dict(d, q=d['q'][segment.start:segment.stop], out=d['out'][segment.start:segment.stop],
                    cu_seqlens_q=segment.cu_query, max_seqlen_q=segment.maximum_query,
                    max_seqlen_k=segment.maximum_kv,
                    # Exact decode lengths stay on the GPU; CPU upper bounds
                    # determine capacity only, never the actual attention mask.
                    seqused_k=d['seqused_k'][segment.first:segment.last],
                    block_table=d['block_table'][segment.first:segment.last],
                    k_descale=d['k_descale'][segment.first:segment.last],
                    v_descale=d['v_descale'][segment.first:segment.last])
                dispatch(original, group, Context(True), capturing=capturing,
                         native_op=native_op, m04=m04, onednn=False)
        return d['out']
    if route == 'M04':
        return verify.run(d)
    return original(**d)


def dispatch(original, d, context, *, capturing, native_op, m04, onednn):
    route = select_route(d, context, capturing=capturing, native_op=native_op, m04=m04, onednn=onednn)
    return _dispatch_selected(route, original, d, context, capturing=capturing, native_op=native_op, m04=m04)


def install():
    if os.environ.get('EXL3_GUARDED_ATTN') != '1':
        return
    import vllm
    from vllm.v1.attention.backends import flash_attn as fa
    from vllm.v1.attention.backend import AttentionType
    if getattr(fa, '_exl3_guarded_attention', False):
        return
    if vllm.__version__ != '0.31.0' or torch.__version__ != '2.14.0+xpu':
        raise RuntimeError('Unsupported guarded attention ABI')
    root = Path(vllm.__file__).parent
    actual = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_HASHES}
    if actual != SOURCE_HASHES:
        raise RuntimeError(f'Unknown guarded attention metadata source: {actual}')
    assert fa.FlashAttentionBackend.forward_includes_kv_cache_update is False
    assert not getattr(fa, '_b70_exl3_m04_installed', False), 'Do not stack attention overlays'
    from .fp8kv_prefill import _onednn_op, L_BUCKET, Q_BUCKET
    onednn = os.environ.get('EXL3_GUARDED_PREFILL') == '1'
    if onednn and (L_BUCKET != 1 or Q_BUCKET != 256):
        raise RuntimeError('Guarded oneDNN requires exact K length and validated query padding')
    native_op = _onednn_op() if onednn else None
    if onednn and native_op is None:
        raise RuntimeError('Requested oneDNN operation unavailable')
    library = os.environ.get('EXL3_M04_LIBRARY')
    if library:
        expected = os.environ['EXL3_M04_LIBRARY_SHA256']
        if hashlib.sha256(Path(library).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Unverified M04 library')
        torch.ops.load_library(library)
        assert torch._C._dispatch_has_kernel_for_dispatch_key('b70_exl3_attention::shared_kv_verify_out', 'XPU')
    original_build = fa.FlashAttentionMetadataBuilder.build
    @functools.wraps(original_build)
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        result = original_build(self, common_prefix_len, common_attn_metadata, fast_build)
        config = self.vllm_config
        spec = config.speculative_config
        fixed = (not (spec and spec.enable_adaptive_verification) and
                 config.parallel_config.decode_context_parallel_size == 1 and
                 config.parallel_config.prefill_context_parallel_size == 1)
        result._exl3_fixed_query_partition = fixed
        # In this reviewed 0.31 source, ordinary fp8 uses load-time scalar
        # scales; dynamic per-token/head formats have different dtype names.
        # The old calculate_kv_scales CacheConfig field no longer exists.
        result._exl3_static_kv_scales = config.cache_config.cache_dtype in ('fp8', 'fp8_e4m3')
        result._exl3_host_plan = metadata.prepare(common_attn_metadata, fixed_partition=fixed) if onednn else None
        return result
    context_var = ContextVar('exl3_guarded_attention', default=Context(False))
    original_forward = fa.FlashAttentionImpl.forward
    @functools.wraps(original_forward)
    def layer_forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                      output_scale=None, output_block_scale=None):
        md = attn_metadata
        allowed = (md is not None and getattr(md, '_exl3_fixed_query_partition', False) and
                   self.attn_type == AttentionType.DECODER and self.dcp_world_size == 1 and
                   self.head_size == 256 and self.num_heads == 24 and self.num_kv_heads == 4 and
                   self.kv_cache_dtype in ('fp8', 'fp8_e4m3') and not md.use_cascade and
                   output_scale is None and output_block_scale is None)
        scales = (getattr(layer, '_k_scale_float', None), getattr(layer, '_v_scale_float', None))
        if (not getattr(md, '_exl3_static_kv_scales', False) or
                any(type(s) not in (int, float) or not math.isfinite(s) or s <= 0 for s in scales)):
            scales = None
        context = Context(allowed, getattr(md, '_exl3_host_plan', None), scales)
        token = context_var.set(context)
        try:
            return original_forward(self, layer, query, key, value, kv_cache, md,
                                    output, output_scale, output_block_scale)
        finally:
            context_var.reset(token)
    original = fa.flash_attn_varlen_func
    seen = set()
    def forward(*args, **d):
        context = context_var.get()
        if args or not context.allowed:
            return original(*args, **d)
        capturing = torch.xpu.is_current_stream_capturing()
        # Log bounded structural evidence once, never per layer/step.
        route = select_route(d, context, capturing=capturing, native_op=native_op, m04=bool(library), onednn=onednn)
        signature = (route, d['cu_seqlens_q'].numel()-1, d['max_seqlen_q'], capturing)
        if signature not in seen and len(seen) < 64:
            seen.add(signature)
            print('EXL3_GUARDED_ATTN_ROUTE', signature, flush=True)
        return _dispatch_selected(route, original, d, context, capturing=capturing,
                                  native_op=native_op, m04=bool(library))
    fa.FlashAttentionMetadataBuilder.build = build
    fa.FlashAttentionImpl.forward = layer_forward
    fa.flash_attn_varlen_func = forward
    fa._exl3_guarded_attention = True
    print('EXL3_GUARDED_ATTN_INSTALL_PASS', actual, 'prefill', onednn, 'M04', bool(library), flush=True)

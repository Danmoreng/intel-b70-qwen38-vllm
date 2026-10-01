"""Experimental shared-KV verification, with native fallback and no KV writes.

Input metadata is the native FA2 contract: max_seqlen_q is the true per-sequence
maximum, cu_seqlens_q describes all actual query rows, and seqused_k includes
the newly written queries. Shape equality then proves uniform query lengths.
This module does not infer lengths by reading device tensors on the host.
"""
import torch

KNOWN = {
    'q', 'k', 'v', 'out', 'cu_seqlens_q', 'max_seqlen_q', 'max_seqlen_k',
    'seqused_k', 'block_table', 'softmax_scale', 'causal', 'window_size',
    'k_descale', 'v_descale', 'fa_version', 'num_splits', 'num_splits_kv',
    'dropout_p', 'softcap', 'return_softmax_lse', 'return_attn_probs',
    'deterministic', 's_aux', 'alibi_slopes', 'q_descale', 'cu_seqlens_k',
    'q_v', 'scheduler_metadata', 'host_kv_lens', 'is_mix_batch',
    'dynamic_causal', 'mask_mod', 'aux_tensors',
}


def eligible(d):
    if set(d) - KNOWN or any(d.get(n) is None for n in
                            ('q', 'k', 'v', 'cu_seqlens_q', 'seqused_k',
                             'block_table', 'k_descale', 'v_descale')):
        return False
    q, k, v = d['q'], d['k'], d['v']
    cq, used, bt = d['cu_seqlens_q'], d['seqused_k'], d['block_table']
    batch, rows = cq.numel() - 1, d.get('max_seqlen_q')
    if not isinstance(rows, int) or not 2 <= rows <= 5 or not 1 <= batch <= 16:
        return False
    if (q.device.type != 'xpu' or q.dtype != torch.float16 or
            tuple(q.shape) != (batch * rows, 24, 256) or not q.is_contiguous()):
        return False
    if (k.dtype != torch.float8_e4m3fn or v.dtype != k.dtype or k.ndim != 4 or
            k.shape[1] <= 0 or k.shape[1] % 64 or tuple(k.shape[2:]) != (4, 256) or
            v.shape != k.shape or v.stride() != k.stride()):
        return False
    strides = k.stride()
    if (strides[3] != 1 or strides[2] < 256 or strides[1] < 4 * strides[2] or
            strides[0] < k.shape[1] * strides[1]):
        return False
    if (cq.ndim != 1 or cq.dtype != torch.int32 or not cq.is_contiguous() or
            used.ndim != 1 or used.numel() != batch or used.dtype != torch.int32 or
            not used.is_contiguous() or bt.ndim != 2 or bt.shape[0] != batch or
            bt.dtype != torch.int32 or not bt.is_contiguous()):
        return False
    maximum = d.get('max_seqlen_k')
    if not isinstance(maximum, int) or not rows <= maximum <= bt.shape[1] * k.shape[1]:
        return False
    if (d.get('causal') is not True or type(d.get('softmax_scale')) not in (int, float) or
            d.get('softmax_scale') != 0.0625):
        return False
    window = d.get('window_size')
    if window is not None and (type(window) not in (tuple, list) or
                              len(window) != 2 or any(type(v) is not int for v in window)):
        return False
    if tuple(window or (-1, -1)) != (-1, -1) or type(d.get('fa_version', 2)) is not int or d.get('fa_version', 2) != 2:
        return False
    for name in ('num_splits', 'num_splits_kv'):
        value = d.get(name)
        if type(value) not in (type(None), int) or value not in (None, 0, 1):
            return False
    for name in ('return_softmax_lse', 'return_attn_probs', 'dropout_p', 'softcap', 'deterministic'):
        flag = d.get(name)
        if type(flag) not in (type(None), bool, int, float) or flag not in (None, False, 0, 0.0):
            return False
    if any(d.get(n) is not None for n in
           ('s_aux', 'alibi_slopes', 'q_descale', 'cu_seqlens_k', 'q_v',
            'scheduler_metadata', 'host_kv_lens', 'dynamic_causal', 'mask_mod', 'aux_tensors')):
        return False
    tensors = [q, k, v, cq, used, bt]
    for key in ('k_descale', 'v_descale'):
        scale = d[key]
        if (scale.dtype != torch.float32 or scale.numel() < 1 or
                scale.untyped_storage().nbytes() != scale.element_size() or
                (scale.numel() != 1 and any(stride != 0 for stride in scale.stride()))):
            return False
        tensors.append(scale)
    out = d.get('out')
    if out is not None:
        if out.shape != q.shape or out.dtype != q.dtype or not out.is_contiguous():
            return False
        tensors.append(out)
    return all(t.device == q.device for t in tensors)


def packed_queries(q, batch, rows):
    return q.view(batch, rows, 4, 6, 256).permute(0, 2, 1, 3, 4).reshape(batch, rows * 24, 256).contiguous()


def unpacked_output(y, batch, rows):
    return y.view(batch, 4, rows, 6, 256).permute(0, 2, 1, 3, 4).reshape(batch * rows, 24, 256).contiguous()


def run(d, splits=None, tile=8):
    batch, rows = d['cu_seqlens_q'].numel() - 1, d['max_seqlen_q']
    q = packed_queries(d['q'], batch, rows)
    y = torch.empty_like(q)
    # Initial historical policy, to be measured independently for C1/C4.
    splits = splits if splits is not None else {2: 32, 3: 8, 4: 16, 5: 16}[rows]
    temp = torch.empty((batch, rows * 24 * splits, 256), dtype=q.dtype, device=q.device)
    sums = torch.empty((batch, rows * 24, splits), dtype=torch.float32, device=q.device)
    maxima = torch.empty_like(sums)
    cu = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
    ks = d['k_descale'].as_strided((1,), (1,))
    vs = d['v_descale'].as_strided((1,), (1,))
    torch.ops.b70_exl3_attention.shared_kv_verify_out(
        q, d['k'], d['v'], d['block_table'], cu, d['seqused_k'], ks, vs,
        y, temp, sums, maxima, d['max_seqlen_k'], splits, tile)
    result = unpacked_output(y, batch, rows)
    out = d.get('out')
    if out is not None:
        out.copy_(result)
        return out
    return result


def dispatch(original, **d):
    if not eligible(d):
        return original(**d)
    return run(d)

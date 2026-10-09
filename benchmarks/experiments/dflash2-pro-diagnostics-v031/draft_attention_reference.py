"""Independent dense oracle for the actual XPU paged DFlash attention ABI."""
import math
import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func as native

RTOL, ATOL = .01, .003  # Frozen before evaluation, same as existing attention gates.


def rotary(x, positions):
    # Independent NeoX RoPE for the checkpoint's full head_dim and theta=1e7.
    frequency = 10000000.**(-torch.arange(0, 128, 2, device=x.device).double()/128)
    angle = positions.double()[:, None]*frequency[None]
    cosine, sine = angle.cos()[:, None], angle.sin()[:, None]
    first, second = x.double().chunk(2, -1)
    return torch.cat((first*cosine-second*sine, second*cosine+first*sine), -1).half()


def dense(d, lengths, ks, vs):
    output = torch.empty_like(d['q'], dtype=torch.float32)
    rows = 8
    for batch, length in enumerate(lengths):
        pages = d['block_table'][batch].long()
        k = d['k'][pages].reshape(-1, 8, 128)[:length].float()*ks
        v = d['v'][pages].reshape(-1, 8, 128)[:length].float()*vs
        q = d['q'][batch*rows:(batch+1)*rows].float()
        q_positions = length-rows+torch.arange(rows, device='xpu')
        keys = torch.arange(length, device='xpu')
        visible = (keys[None]-q_positions[:, None]).abs() <= 2047
        # GQA=4: each KV head serves four Q heads. No causal triangle.
        q = q.view(rows, 8, 4, 128).permute(1, 2, 0, 3)
        scores = torch.matmul(q, k.permute(1, 2, 0)[:, None])*128**-.5
        scores.masked_fill_(~visible[None, None], -math.inf)
        result = torch.matmul(scores.softmax(-1), v.permute(1, 0, 2)[:, None])
        output[batch*rows:(batch+1)*rows] = result.permute(2, 0, 1, 3).reshape(rows, 32, 128)
    return output


def compare(actual, expected):
    error = (actual.float()-expected.float()).abs()
    result = dict(max_abs=float(error.max().cpu()), rms=float(error.square().mean().sqrt().cpu()),
                  finite=bool(torch.isfinite(actual).all().cpu()),
                  allclose=bool(torch.allclose(actual.float(), expected.float(), rtol=RTOL, atol=ATOL)))
    assert result['finite'] and result['allclose'], result
    return result


def gpu_test():
    torch.manual_seed(20261009)
    checks = []
    for page in (64, 1664):
        for batch in (1, 4):
            for length in sorted({8, page-1, page, page+1, 2047, 2048, 2049}):
                page_count = (length+7+page-1)//page
                # Last physical pages contain invalid/stale tail slots.
                original = torch.randn(batch*page_count, page, 8, 2, 128, device='xpu', dtype=torch.float16)*.7
                tables = torch.randperm(batch*page_count, device='xpu').reshape(batch, page_count).int()
                q = torch.randn(batch*8, 32, 128, device='xpu', dtype=torch.float16)
                for b in range(batch):
                    qp = 33000+torch.arange(length-8, length, device='xpu')
                    q[b*8:(b+1)*8] = rotary(q[b*8:(b+1)*8], qp)
                    for logical in range(page_count):
                        physical = tables[b, logical].long()
                        kp = 33000+logical*page+torch.arange(page, device='xpu')
                        original[physical, :, :, 0] = rotary(original[physical, :, :, 0], kp)
                for dtype in (torch.float8_e4m3fn, torch.float16):
                    ks, vs = (.75, 1.25) if dtype == torch.float8_e4m3fn else (1., 1.)
                    base = original.clone()
                    base[:, :, :, 0] /= ks; base[:, :, :, 1] /= vs
                    base = base.to(dtype)
                    lengths = [length]*batch
                    d = dict(q=q, k=base[:, :, :, 0], v=base[:, :, :, 1],
                        out=torch.empty_like(q), cu_seqlens_q=torch.arange(batch+1, device='xpu', dtype=torch.int32)*8,
                        seqused_k=torch.tensor(lengths, device='xpu', dtype=torch.int32),
                        block_table=tables, max_seqlen_q=8, max_seqlen_k=length+7,
                        softmax_scale=128**-.5, causal=False, window_size=(2047, 2047))
                    if dtype == torch.float8_e4m3fn:
                        d.update(k_descale=torch.tensor(ks, device='xpu').expand(batch, 8),
                                 v_descale=torch.tensor(vs, device='xpu').expand(batch, 8))
                    actual = native(**d)
                    reference = dense(d, lengths, ks, vs)
                    row = dict(page=page, batch=batch, length=length, dtype=str(dtype),
                               same_stored_kv=compare(actual, reference))
                    before = actual.clone()
                    # Poison invalid tail slots; seqused_k must prevent reads.
                    for b in range(batch):
                        for pos in range(length, min(length+7, page_count*page)):
                            physical = tables[b, pos//page].long()
                            d['v'][physical, pos%page].fill_(224)
                    poisoned = native(**d)
                    assert torch.equal(before, poisoned), 'Invalid/rejected tail slots leaked'
                    row['stale_tail_bitexact'] = True
                    if length == 2049:
                        # First query excludes the oldest key; the last query
                        # includes all eight future block rows (noncausal).
                        for _ in range(3): native(**d)
                        torch.xpu.synchronize(); graph = torch.xpu.XPUGraph()
                        with torch.xpu.graph(graph): graph_output = native(**d)
                        graph.replay(); row['graph'] = compare(graph_output, dense(d, lengths, ks, vs))
                        lengths = [length-b-1 for b in range(batch)]
                        d['seqused_k'].copy_(torch.tensor(lengths, device='xpu', dtype=torch.int32))
                        d['block_table'].copy_(d['block_table'].flip(1))
                        d['q'].mul_(.5); graph.replay()
                        row['changed_length_pages_query_graph'] = compare(graph_output, dense(d, lengths, ks, vs))
                    checks.append(row)
    return dict(status='PASS_DFLASH_PAGED_NATIVE_ATTENTION_DENSE_REFERENCE',
        tolerance=dict(rtol=RTOL, atol=ATOL), checks=checks,
        geometry=dict(query_rows=8, query_heads=32, kv_heads=8, head_dim=128,
                      causal=False, sliding_window_left=2047, sliding_window_right=2047, rope_theta=10000000),
        scope='Synthetic paged KV and independent dense FP32 oracle for same stored FP8 or FP16 values, stale tails, C1/C4 and graph mutation. Independent RoPE-generated inputs; does not yet compare runtime RoPE implementation or replay captured model projections.')

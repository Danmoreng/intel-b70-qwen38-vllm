"""Exact-K mixed dispatch numerical/causal/graph gates on the actual XPU ABI."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func as native

from exl3xpu.attention_metadata import prepare
from exl3xpu.attention_dispatch import Context, dispatch
from exl3xpu.fp8kv_prefill import _onednn_op

RTOL, ATOL = .01, .003


def make(queries, lengths, prefilling, page=1600):
    n = len(queries)
    pages = (max(lengths)+page-1)//page
    storage = torch.randn((n*pages, page, 4, 2, 256), device='xpu', dtype=torch.float16).to(torch.float8_e4m3fn)
    table = torch.randperm(n*pages, device='xpu').view(n, pages).int()
    bounds = [0]
    for q in queries: bounds.append(bounds[-1]+q)
    upper = [length if flag else length+7 for length, flag in zip(lengths, prefilling)]
    common = SimpleNamespace(max_query_len=max(queries), num_reqs=n, num_actual_tokens=sum(queries),
        query_start_loc_cpu=torch.tensor(bounds, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor(upper, dtype=torch.int32),
        is_prefilling=torch.tensor(prefilling, dtype=torch.bool),
        query_start_loc=torch.tensor(bounds, device='xpu', dtype=torch.int32))
    d = dict(q=torch.randn((sum(queries), 24, 256), device='xpu', dtype=torch.float16),
        k=storage[:, :, :, 0], v=storage[:, :, :, 1],
        out=torch.empty((sum(queries), 24, 256), device='xpu', dtype=torch.float16),
        cu_seqlens_q=common.query_start_loc, seqused_k=torch.tensor(lengths, device='xpu', dtype=torch.int32),
        block_table=table, max_seqlen_q=max(queries), max_seqlen_k=max(upper),
        softmax_scale=.0625, causal=True,
        k_descale=torch.tensor(.75, device='xpu').expand(n, 4),
        v_descale=torch.tensor(1.25, device='xpu').expand(n, 4))
    return d, Context(True, prepare(common, fixed_partition=True), (.75, 1.25))


def dense(d):
    boundaries = d['cu_seqlens_q'].cpu().tolist()
    result = torch.empty_like(d['q'], dtype=torch.float32)
    for b, length in enumerate(d['seqused_k'].cpu().tolist()):
        start, stop = boundaries[b:b+2]
        q = d['q'][start:stop].float()
        k = d['k'][d['block_table'][b].long()].reshape(-1, 4, 256)[:length].float()*.75
        v = d['v'][d['block_table'][b].long()].reshape(-1, 4, 256)[:length].float()*1.25
        positions = torch.arange(length, device='xpu')
        visible = length-(stop-start)+torch.arange(stop-start, device='xpu')[:, None]
        for h in range(24):
            scores = q[:, h]@k[:, h//6].T*.0625
            scores.masked_fill_(positions[None] > visible, -float('inf'))
            result[start:stop, h] = scores.softmax(-1)@v[:, h//6]
    return result


def check(a, b):
    a, b = a.float(), b.float()
    error = (a-b).abs()
    row = dict(allclose=bool(torch.allclose(a,b,rtol=RTOL,atol=ATOL)),
               finite=bool(a.isfinite().all()), maximum_absolute=float(error.max()),
               rms=float(error.square().mean().sqrt()))
    assert row['allclose'] and row['finite'], row
    return row


def reference(d):
    # Native must not overwrite the captured/candidate output under test.
    return native(**dict(d, out=torch.empty_like(d['q'])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--library', required=True); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); assert not args.out.exists(); args.out.mkdir(parents=True)
    torch.manual_seed(20261001); torch.ops.load_library(args.library)
    op = _onednn_op(); assert op is not None
    calls = []
    class CountedOp:
        def exl3_sdpa_len(self, *args):
            calls.append((args[-2], args[-1]))
            return op.exl3_sdpa_len(*args)
    counted = CountedOp()
    result = dict(status='RUNNING', tolerance=dict(rtol=RTOL,atol=ATOL), cases=[], graphs=[], causal=[])
    def save(): (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    def run(d, context, capture=False):
        return dispatch(native, d, context, capturing=capture, native_op=counted, m04=True, onednn=True)
    save()
    try:
        cases = [([q], [length], [True], 1600)
                 for q in (63,64,255,256) for length in (4095,4096,4097)]
        cases += [([4,4,64,3], [1601,1701,4097,3201], [False,False,True,False], page)
                  for page in (64,1600,1664,1792)]
        cases += [([256,4,4,64], [8193,8191,9000,4097], [True,False,False,True], 1600),
                  ([64]*4, [4096,4097,8191,8193], [True]*4, 1600)]
        cases += [([q], [length], [False], 1600) for q in (1,2,3,4,5) for length in (1600,1601)]
        cases += [([64],[length],[True],page) for page in (1600,1664,1792) for length in (32769,102753)]
        for queries,lengths,flags,page in cases:
            d, context = make(queries,lengths,flags,page)
            before = d['k'].view(torch.uint8).clone(), d['v'].view(torch.uint8).clone()
            expected = reference(d).clone(); before_calls = len(calls)
            actual = run(d,context).clone()
            expected_calls = 4*sum(s.prefill for s in context.plan.segments) if context.plan else 0
            assert len(calls)-before_calls == expected_calls
            row = dict(queries=queries,lengths=lengths,prefilling=flags,page=page,
                       planned=context.plan is not None, native=check(actual,expected),
                       onednn_calls=len(calls)-before_calls)
            if max(lengths) <= 8193: row['fp32'] = check(actual,dense(d))
            row['kv_unchanged'] = bool(torch.equal(before[0],d['k'].view(torch.uint8)) and
                                       torch.equal(before[1],d['v'].view(torch.uint8)))
            assert row['kv_unchanged']; result['cases'].append(row); save()
            print('PASS',queries,lengths,page,'plan',context.plan is not None,flush=True)
        for q in (64,255,256):
            d, context = make([q],[4097],[True])
            original = run(d,context).clone()
            for position in range(4097-q+1,4097):
                physical = d['block_table'][0,position//1600].long()
                d['v'][physical,position%1600].fill_(224)
            actual = run(d,context)
            assert torch.equal(original[0],actual[0]), 'future KV leaked'
            assert not torch.equal(original[-1],actual[-1]), 'poison control ineffective'
            result['causal'].append(dict(q=q,first_row_bitexact=True,last_row_changes=True)); save()
        for queries,lengths,flags in [([4]*4,[4096]*4,[False]*4),([64],[4097],[True])]:
            d, context = make(queries,lengths,flags)
            before_calls = len(calls)
            for _ in range(3): run(d,context,True)
            torch.xpu.synchronize(); graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph): actual = run(d,context,True)
            graph.replay(); row = dict(queries=queries,initial=check(actual,reference(d).clone()))
            d['q'].mul_(.5); d['block_table'].copy_(d['block_table'].flip(1)); d['seqused_k'].sub_(1)
            graph.replay(); row['mutated'] = check(actual,reference(d).clone())
            assert len(calls) == before_calls, 'oneDNN entered graph path'
            result['graphs'].append(row); save()
        result['status'] = 'PASS_ATTENTION_MICRO_GATES_SERVING_UNQUALIFIED'
    except BaseException as exc:
        result['status']='FAILED'; result['error']=repr(exc); raise
    finally: save()


if __name__ == '__main__': main()

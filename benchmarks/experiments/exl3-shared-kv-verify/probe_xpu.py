"""Actual XPU eager/graph/causal guards and isolated C1/C4 attention timing."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func as native
import verify_attention as V

RTOL, ATOL = 0.01, 0.003  # Existing M04 tolerance, fixed before these probes.


def delta(actual, expected):
    error = (actual.float() - expected.float()).abs()
    result = {'finite': bool(actual.isfinite().all().cpu()),
              'allclose': bool(torch.allclose(actual.float(), expected.float(), rtol=RTOL, atol=ATOL)),
              'max_abs': float(error.max().cpu()), 'rms': float(error.square().mean().sqrt().cpu())}
    assert result['finite'] and result['allclose'], result
    return result


def make(batch, rows, page, maximum):
    pages = (maximum + page - 1) // page
    base = torch.randn((batch * pages, page, 4, 2, 256), dtype=torch.float16, device='xpu').to(torch.float8_e4m3fn)
    # Separate randomized pages for each sequence; no accidental C4 aliasing.
    tables = torch.randperm(batch * pages, device='xpu').view(batch, pages).int()
    lengths = [maximum - b * min(7, maximum - rows) for b in range(batch)]
    lengths = [max(rows, v) for v in lengths]
    d = dict(q=torch.randn((batch * rows, 24, 256), device='xpu', dtype=torch.float16),
             k=base[:, :, :, 0], v=base[:, :, :, 1],
             cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32, device='xpu') * rows,
             seqused_k=torch.tensor(lengths, dtype=torch.int32, device='xpu'),
             block_table=tables, max_seqlen_q=rows, max_seqlen_k=maximum,
             softmax_scale=0.0625, causal=True,
             k_descale=torch.tensor(0.75, dtype=torch.float32, device='xpu').expand(batch, 4),
             v_descale=torch.tensor(1.25, dtype=torch.float32, device='xpu').expand(batch, 4))
    return d, lengths


def dense(d, lengths):
    batch, rows = len(lengths), d['max_seqlen_q']
    gold = torch.empty_like(d['q'], dtype=torch.float32)
    for b, length in enumerate(lengths):
        k = d['k'][d['block_table'][b].long()].reshape(-1, 4, 256)[:length].float() * 0.75
        v = d['v'][d['block_table'][b].long()].reshape(-1, 4, 256)[:length].float() * 1.25
        q = d['q'][b * rows:(b + 1) * rows].float()
        positions = torch.arange(length, device='xpu')
        for head in range(24):
            scores = q[:, head] @ k[:, head // 6].T * 0.0625
            visible = length - rows + torch.arange(rows, device='xpu')[:, None]
            scores.masked_fill_(positions[None] > visible, -float('inf'))
            gold[b * rows:(b + 1) * rows, head] = scores.softmax(-1) @ v[:, head // 6]
    return gold


def timing(function):
    start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    torch.xpu.synchronize(); host = time.perf_counter(); start.record()
    function(); end.record(); end.synchronize()
    return {'event_ms': start.elapsed_time(end), 'wall_ms': (time.perf_counter() - host) * 1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); assert not args.out.exists(); args.out.mkdir()
    torch.ops.load_library(str(args.library)); torch.manual_seed(20261001)
    result = {'status':'RUNNING','torch':torch.__version__,
              'library_sha256':hashlib.sha256(args.library.read_bytes()).hexdigest(),
              'tolerance':{'rtol':RTOL,'atol':ATOL}, 'checks':[], 'graphs':[], 'fallbacks':[], 'timings':[]}
    def save(): (args.out / 'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        for page in (64, 1600, 1664):
            for batch in (1, 4):
                for rows in (2, 3, 4, 5):
                    for length in (rows, 65, page - 1, page, page + 1):
                        d, lengths = make(batch, rows, page, length)
                        assert V.eligible(d)
                        before = d['k'].view(torch.uint8).clone(), d['v'].view(torch.uint8).clone()
                        reference = native(**d)
                        actual = V.dispatch(native, **d)
                        row = {'page':page,'batch':batch,'q':rows,'kv_lens':lengths,
                               'native':delta(actual,reference),'fp32':delta(actual,dense(d,lengths)),
                               'kv_unchanged':bool(torch.equal(before[0],d['k'].view(torch.uint8)) and torch.equal(before[1],d['v'].view(torch.uint8)))}
                        assert row['kv_unchanged']; result['checks'].append(row); save()
                    print(f'PASS eager page{page} C{batch} q{rows}',flush=True)
        # Future-token poison: row0 must not see any of the last q-1 values.
        for rows in (2,3,4,5):
            d,_=make(1,rows,1600,1601); original=V.run(d).clone()
            for pos in range(1601-rows+1,1601):
                page_id=d['block_table'][0,pos//1600].long()
                d['v'][page_id,pos%1600].fill_(224)
            poisoned=V.run(d)
            assert torch.equal(original[0],poisoned[0]),'future KV leaked into first query'
            assert not torch.equal(original[-1],poisoned[-1]),'poison control did not affect visible row'
            result['checks'].append({'causal_poison_q':rows,'first_row_bitexact':True,'last_row_changes':True});save()
        for page in (1600,1664):
            for batch in (1,4):
                for rows in (2,4,5):
                    d,lengths=make(batch,rows,page,page+1)
                    for _ in range(3): V.run(d)
                    torch.xpu.synchronize(); graph=torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph): output=V.run(d)
                    graph.replay(); eager=V.run(d); check=delta(output,eager)
                    # Replay must consume updated device lengths and block pages.
                    d['block_table'].copy_(d['block_table'].flip(1)); lengths=[v-1 for v in lengths]
                    d['seqused_k'].copy_(torch.tensor(lengths,dtype=torch.int32,device='xpu'))
                    d['q'].mul_(0.5); graph.replay()
                    row={'page':page,'batch':batch,'q':rows,'initial':check,
                         'mutated_native':delta(output,native(**d)), 'mutated_fp32':delta(output,dense(d,lengths))}
                    result['graphs'].append(row);save()
        for batch in (1,4):
            for rows in (1,6,63,64,255,256):
                d,_=make(batch,rows,1600,1601); assert not V.eligible(d)
                expected=native(**d); actual=V.dispatch(native,**d)
                result['fallbacks'].append({'batch':batch,'q':rows,'native':delta(actual,expected)});save()
        for batch in (1,4):
            for rows in (2,3,4,5):
                for length in (8192,102752,196608):
                    d,_=make(batch,rows,1600,length); expected=native(**d)
                    result['checks'].append({'batch':batch,'q':rows,'kv':length,'long_native':delta(V.run(d),expected)});save()
                    candidates={'native':lambda:native(**d),'m04':lambda:V.run(d)}
                    for fn in candidates.values():
                        for _ in range(3): fn()
                    samples={k:[] for k in candidates}; rng=random.Random(batch+rows+length)
                    for _ in range(9):
                        order=list(candidates);rng.shuffle(order)
                        for name in order:samples[name].append(timing(candidates[name]))
                    med={k:statistics.median(v['event_ms'] for v in values) for k,values in samples.items()}
                    row={'batch':batch,'q':rows,'kv':length,'median_event_ms':med,
                         'speedup_pct':100*(med['native']/med['m04']-1),'samples':samples}
                    result['timings'].append(row);save();print('TIMING',batch,rows,length,med,flush=True)
        result['status']='COMPLETE_MICRO_GATES_ONLY_SERVING_UNQUALIFIED'
    except BaseException as exc:
        result['status']='FAILED';result['error']=repr(exc);raise
    finally:save()


if __name__=='__main__':main()

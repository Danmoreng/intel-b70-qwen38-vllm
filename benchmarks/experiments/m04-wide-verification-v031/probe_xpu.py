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
from exl3xpu import shared_kv_verify as V
from exl3xpu.attention_dispatch import Context, select_route

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
             out=torch.empty((batch * rows, 24, 256), device='xpu', dtype=torch.float16),
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


def native_reference(d):
    return native(**dict(d, out=torch.empty_like(d['q'])))


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
                for rows in range(2, 10):
                    for length in (rows, 65, page - 1, page, page + 1):
                        d, lengths = make(batch, rows, page, length)
                        assert V.eligible(d)
                        before = d['k'].view(torch.uint8).clone(), d['v'].view(torch.uint8).clone()
                        reference = native_reference(d)
                        actual = V.dispatch(native, **d)
                        row = {'page':page,'batch':batch,'q':rows,'kv_lens':lengths,
                               'native':delta(actual,reference),'fp32':delta(actual,dense(d,lengths)),
                               'kv_unchanged':bool(torch.equal(before[0],d['k'].view(torch.uint8)) and torch.equal(before[1],d['v'].view(torch.uint8)))}
                        assert row['kv_unchanged']; result['checks'].append(row); save()
                    print(f'PASS eager page{page} C{batch} q{rows}',flush=True)
        # Future-token poison must not leak through either packed-Q tile.
        for rows in range(2,10):
            for tile in ((8,16) if rows>=6 else (8,)):
                d,lengths=make(1,rows,1600,1601); original=V.run(d,tile=tile).clone()
                delta(original,dense(d,lengths))
                for pos in range(1601-rows+1,1601):
                    page_id=d['block_table'][0,pos//1600].long()
                    d['v'][page_id,pos%1600].fill_(224)
                poisoned=V.run(d,tile=tile)
                assert torch.equal(original[0],poisoned[0]),'future KV leaked into first query'
                assert not torch.equal(original[-1],poisoned[-1]),'poison control did not affect visible row'
                result['checks'].append({'causal_poison_q':rows,'tile':tile,'first_row_bitexact':True,'last_row_changes':True});save()
        for page in (1600,1664):
            for batch in (1,4):
                for rows in (2,4,5,6,7,8,9):
                    for tile in ((8,16) if rows>=6 else (8,)):
                        d,lengths=make(batch,rows,page,page+1)
                        for _ in range(3): V.run(d,tile=tile)
                        torch.xpu.synchronize(); graph=torch.xpu.XPUGraph()
                        with torch.xpu.graph(graph): output=V.run(d,tile=tile)
                        graph.replay(); check=delta(output,native_reference(d))
                        # Replay consumes changed device lengths, pages and Q.
                        d['block_table'].copy_(d['block_table'].flip(1)); lengths=[v-1 for v in lengths]
                        d['seqused_k'].copy_(torch.tensor(lengths,dtype=torch.int32,device='xpu'))
                        d['q'].mul_(0.5); graph.replay()
                        row={'page':page,'batch':batch,'q':rows,'tile':tile,'initial':check,
                             'mutated_native':delta(output,native_reference(d)), 'mutated_fp32':delta(output,dense(d,lengths))}
                        result['graphs'].append(row);save()
        for batch in (1,4):
            for rows in (1,10,63,64,255,256):
                d,_=make(batch,rows,1600,1601); assert not V.eligible(d)
                expected=native_reference(d); actual=V.dispatch(native,**d)
                result['fallbacks'].append({'batch':batch,'q':rows,'native':delta(actual,expected)});save()
        # Actual GPU lengths must select new splits independently of the
        # fixed maximum captured on the host, including heterogeneous C4.
        for rows in (8,9):
            d,_=make(4,rows,1664,96000)
            initial=[65,8193,49153,96000]
            d['seqused_k'].copy_(torch.tensor(initial,dtype=torch.int32,device='xpu'))
            for _ in range(3):V.run(d)
            torch.xpu.synchronize();graph=torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):output=V.run(d)
            graph.replay();first=delta(output,native_reference(d));delta(output,dense(d,initial))
            changed=[96000,4096,32768,65536]
            d['seqused_k'].copy_(torch.tensor(changed,dtype=torch.int32,device='xpu'))
            d['block_table'].copy_(d['block_table'].flip(1));d['q'].mul_(.5)
            graph.replay()
            result['graphs'].append(dict(q=rows,adaptive_heterogeneous_C4=True,initial_lengths=initial,
                                         replay_lengths=changed,initial=first,mutated_native=delta(output,native_reference(d)),
                                         mutated_fp32=delta(output,dense(d,changed))))
            save()
        # Tune only the new sizes; preserve existing q2..5 serving policy.
        for batch in (1,4):
            for rows in (6,7,8,9):
                for length in (4096,32768,65536,96000):
                    d,_=make(batch,rows,1664,length); expected=native_reference(d)
                    candidates={'native':lambda:native(**d), 'adaptive':lambda:V.run(d)}
                    result['checks'].append({'batch':batch,'q':rows,'kv':length,'adaptive_native':delta(V.run(d),expected)})
                    for tile in (8,16):
                        for splits in (1,4,8,16,32):
                            name=f'q{tile}-s{splits}'
                            fn=lambda splits=splits,tile=tile:V.run(d,splits=splits,tile=tile)
                            result['checks'].append({'batch':batch,'q':rows,'kv':length,'tile':tile,'splits':splits,'long_native':delta(fn(),expected)})
                            candidates[name]=fn
                    for fn in candidates.values():
                        for _ in range(3): fn()
                    samples={k:[] for k in candidates}; rng=random.Random(batch+rows+length)
                    for _ in range(9):
                        order=list(candidates);rng.shuffle(order)
                        for name in order:samples[name].append(timing(candidates[name]))
                    med={k:statistics.median(v['event_ms'] for v in values) for k,values in samples.items()}
                    best=min((name for name in med if name!='native'),key=med.get)
                    row={'batch':batch,'q':rows,'kv':length,'median_event_ms':med,
                         'selected':best,'speedup_pct':100*(med['native']/med[best]-1),'samples':samples}
                    result['timings'].append(row);save();print('TIMING',batch,rows,length,best,med['native'],med[best],flush=True)
        # Preserve production concurrency and feature guards.
        for batch,rows in [(1,8),(4,9),(5,8),(16,9)]:
            d,_=make(batch,rows,1664,1665)
            route=select_route(d,Context(True),capturing=False,native_op=None,m04=True,onednn=False)
            assert route==('M04' if batch<=4 else 'native')
            result['fallbacks'].append(dict(batch=batch,q=rows,checked_route=route));save()
        for change in [dict(causal=False),dict(window_size=(2047,0)),dict(softcap=1.0),dict(unknown_feature=True)]:
            d,_=make(1,8,1664,1665);d.update(change)
            assert select_route(d,Context(True),capturing=False,native_op=None,m04=True,onednn=False)=='native'
            result['fallbacks'].append(dict(unsupported_features=change,checked_route='native'));save()
        result['status']='COMPLETE_MICRO_GATES_ONLY_SERVING_UNQUALIFIED'
    except BaseException as exc:
        result['status']='FAILED';result['error']=repr(exc);raise
    finally:save()


if __name__=='__main__':main()

"""oneDNN SDPA host-read removal: numerical, scale, stream and thread gates."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
from exl3xpu import ops


def compare(actual, gold):
    error=(actual.float()-gold.float()).abs()
    result={'finite':bool(actual.isfinite().all().cpu()),
            'allclose':bool(torch.allclose(actual.float(),gold.float(),rtol=0.01,atol=0.003)),
            'max_abs':float(error.max().cpu())}
    assert result['finite'] and result['allclose'],result
    return result


def inputs(rows,length,pad=256):
    q=torch.randn((6,pad,256),dtype=torch.float16,device='xpu')
    k=torch.randn((1,length,256),dtype=torch.float16,device='xpu')
    v=torch.randn_like(k);out=torch.empty_like(q);none=torch.empty(0,device='xpu')
    return q,k,v,out,none


def gold(q,k,v,rows,scale):
    reciprocal=torch.tensor(1.0/scale,dtype=torch.float16).float()
    scores=q[:,-rows:].float() @ k[0].float().T / float(reciprocal)
    visible=k.shape[1]-rows+torch.arange(rows,device='xpu')[:,None]
    keys=torch.arange(k.shape[1],device='xpu')
    scores.masked_fill_(keys[None]>visible,-float('inf'))
    return scores.softmax(-1) @ v[0].float()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--baseline',action='store_true',help='Measure old scalar reads; skip new invalid-scale/thread guarantees')
    args=p.parse_args();assert not args.report.exists()
    torch.manual_seed(20261001);E=ops._get_esimd()
    result={'status':'RUNNING','baseline':args.baseline,'torch':torch.__version__,'checks':[],
            'library_sha256':hashlib.sha256(Path(os.environ.get('EXL3_LIB',Path(ops.__file__).parent/'_C.so')).read_bytes()).hexdigest()}
    def save():args.report.write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        for rows in (63,64,255,256):
            for length in (1600,1601,4096):
                q,k,v,out,none=inputs(rows,length)
                for scale in (0.0625,0.125,0.09375,0.0625):
                    E.exl3_sdpa_len(q,k,v,out,none,True,scale,length,256)
                    result['checks'].append({'rows':rows,'length':length,'scale':scale,
                                             'fp32':compare(out[:,-rows:],gold(q,k,v,rows,scale))})
                before=out[:,-rows:].clone();q[:,:256-rows].fill_(32)
                E.exl3_sdpa_len(q,k,v,out,none,True,0.0625,length,256)
                assert torch.equal(before,out[:,-rows:]),'padded queries changed valid output'
                save()
        q,k,v,out,none=inputs(256,65537)
        E.exl3_sdpa(q,k,v,out,none,True,0.0625)
        result['checks'].append({'rows':256,'length':65537,'scale':0.0625,
                                 'fp32':compare(out,gold(q,k,v,256,0.0625))});save()
        q,k,v,out,none=inputs(256,4096)
        fn=lambda:E.exl3_sdpa(q,k,v,out,none,True,0.0625)
        for _ in range(3):fn()
        torch.xpu.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            for _ in range(9):
                for _ in range(4):fn()  # Same four-head loop as prefill caller.
        torch.xpu.synchronize()
        counts={event.key:event.count for event in profiler.key_averages()}
        reads=counts.get('aten::_local_scalar_dense',0)
        result['profile_cpu_op_counts']=counts;result['device_scalar_reads_in36_calls']=reads
        assert reads==(36 if args.baseline else 0),counts
        samples=[]
        for _ in range(9):
            start,end=torch.xpu.Event(enable_timing=True),torch.xpu.Event(enable_timing=True)
            torch.xpu.synchronize();host=time.perf_counter();start.record()
            for _ in range(4):fn()
            end.record();end.synchronize()
            samples.append({'event_ms':start.elapsed_time(end),'wall_ms':(time.perf_counter()-host)*1000})
        result['four_head_samples']=samples
        result['median_four_head_ms']={k:statistics.median(v[k] for v in samples) for k in ['event_ms','wall_ms']}
        if not args.baseline:
            result['invalid_scales']=[]
            for scale in (0.0,-0.0625,float('nan'),float('inf'),-float('inf'),1e-30,1e300):
                try:E.exl3_sdpa(q,k,v,out,none,True,scale)
                except RuntimeError as exc:
                    assert 'scale' in str(exc);result['invalid_scales'].append(repr(scale))
                else:raise AssertionError(f'Invalid scale accepted: {scale}')
            # Independent streams/threads create separate engines, partitions
            # and constants. Source input tensors are synchronized before use.
            cases=[inputs(64,1601),inputs(64,4096)];torch.xpu.synchronize()
            def worker(index):
                torch.xpu.set_device(0);stream=torch.xpu.Stream()
                with torch.xpu.stream(stream):
                    q,k,v,out,none=cases[index];checks=[]
                    for scale in (0.0625,0.125,0.0625):
                        E.exl3_sdpa_len(q,k,v,out,none,True,scale,k.shape[1],256)
                        checks.append(compare(out[:,-64:],gold(q,k,v,64,scale)))
                    stream.synchronize();return checks
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                result['thread_stream_checks']=list(pool.map(worker,[0,1]))
        result['status']='PASS_NUMERICAL_AND_HOST_READ_GATES_SERVING_UNQUALIFIED'
    except BaseException as exc:result['status']='FAILED';result['error']=repr(exc);raise
    finally:save()
    print(json.dumps({'status':result['status'],'device_scalar_reads':reads,'median_four_head_ms':result['median_four_head_ms']}))


if __name__=='__main__':main()

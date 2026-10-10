"""Original M04 contract: adversarial lengths, stateful graph replay and timings."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from exl3xpu.shared_kv_verify import run,eligible
import sys
sys.path.insert(0,'/opt/exl3xpu/tests')
from test_guarded_attention_xpu import make,check,reference,dense

def digest(t):return hashlib.sha256(t.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def timed(fn,iterations=100):
    for _ in range(10):fn()
    torch.xpu.synchronize();start=time.perf_counter()
    for _ in range(iterations):fn()
    torch.xpu.synchronize();return (time.perf_counter()-start)*1000/iterations
def main():
    p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.ops.load_library(a.library)
    report=dict(status='RUNNING',library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),cases=[],
        tolerance=dict(rtol=.01,atol=.003),scope='M04 whole-call graph microbenchmark, not serving throughput',
        timing_note='Graph replay host wall time with synchronization outside timed loop; no profiler. Full pack/reduce/output-copy included.')
    def save():(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for batch in [1,4]:
            for rows in [2,3,4,5]:
                for length in [4097,32769,65537,131073,261121]:
                    torch.manual_seed(871100+batch*100+rows*10)
                    # Short siblings exercise inactive/empty splits beside a
                    # long request. max_seqlen_k is a bound, device lengths exact.
                    lengths=[length] if batch==1 else [length,max(rows, length-1599),4097,1601]
                    d,context=make([rows]*batch,lengths,[False]*batch,1600)
                    assert eligible(d)
                    result=run(d).clone();row=dict(batch=batch,rows=rows,lengths=lengths,
                        native=check(result,reference(d)),output_sha256=digest(result))
                    if length==4097:row['fp32']=check(result,dense(d))
                    # Capture actual computation and change both inputs and
                    # device-side lengths before replay, not frozen outputs.
                    stream=torch.xpu.Stream()
                    with torch.xpu.stream(stream):
                        for _ in range(3):run(d)
                    stream.synchronize();graph=torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph,stream=stream):run(d)
                    torch.xpu.synchronize();row['graph_ms']=timed(graph.replay)
                    d['q'].mul_(.75)
                    d['seqused_k'].sub_(1)
                    # Change page lookup as well, preserving allocated bounds.
                    d['block_table'].copy_(d['block_table'].roll(1,dims=1))
                    expected=reference(d).clone();graph.replay();torch.xpu.synchronize()
                    row['mutated_graph']=check(d['out'],expected)
                    report['cases'].append(row);save();print('M04_PASS',batch,rows,length,row['graph_ms'],flush=True)
                    del graph,d,context,result,expected;torch.xpu.empty_cache()
        report['status']='PASS_M04_ORIGINAL_CONTRACT_AND_MUTATED_GRAPH'
    except BaseException as e:report.update(status='FAILED',error=repr(e));raise
    finally:save()
if __name__=='__main__':main()

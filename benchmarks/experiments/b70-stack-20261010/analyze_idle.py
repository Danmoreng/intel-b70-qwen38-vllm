"""Union real timeline intervals; never label profiler gaps as host idle proof."""
import argparse,hashlib,json,re,statistics
from collections import defaultdict
from pathlib import Path
def union_length(intervals):
    result=0;left=right=None
    for a,b in sorted(intervals):
        if left is None:left,right=a,b
        elif a<=right:right=max(right,b)
        else:result+=right-left;left,right=a,b
    return result+(right-left if left is not None else 0)
def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();reports={}
    for file in a.root.glob('*-trace.json'):
        events=json.loads(file.read_text())['traceEvents'];annotations=defaultdict(list)
        for e in events:
            if e.get('cat')=='gpu_user_annotation':
                m=re.match(r'exl3_diagnostic/cycle(\d+)/',e['name'])
                if m:annotations[int(m.group(1))].append(e)
        work=[e for e in events if e.get('cat') in ('kernel','gpu_memcpy') and e.get('dur',0)>0]
        rows=[]
        for cycle,marks in sorted(annotations.items()):
            lo=min(e['ts'] for e in marks);hi=max(e['ts']+e['dur'] for e in marks)
            spans=[(max(lo,e['ts']),min(hi,e['ts']+e['dur'])) for e in work if e['ts']<hi and e['ts']+e['dur']>lo]
            busy=union_length(spans)
            rows.append(dict(cycle=cycle,annotation_span_ms=(hi-lo)/1000,visible_kernel_memcpy_union_ms=busy/1000,
                unaccounted_interval_ms=(hi-lo-busy)/1000,visible_work_fraction=busy/(hi-lo)))
        reports[file.name]=dict(trace_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),cycles=rows,
            median_visible_work_fraction=statistics.median(x['visible_work_fraction'] for x in rows))
    d=dict(status='DIAGNOSTIC_TIMELINE_ONLY_NO_HOST_PATCH_JUSTIFIED',reports=reports,
        limitations=['Union avoids double-counting simultaneous kernels across queues.',
            'Annotation envelope is not a calibrated whole-round critical path.',
            'Unaccounted intervals include barriers, device graph dependencies, scheduling and profiler overhead; they do not prove avoidable host idle.',
            'Engine scheduler and client live in other processes and are not fully covered by worker Kineto.',
            'Trace waves are slower than uninstrumented controls; no async or graph-policy retry justified from this subtraction.'],
        E06_decision='INCONCLUSIVE_AVOIDABLE_HOST_OPERATION_NOT_IDENTIFIED')
    a.out.write_text(json.dumps(d,indent=2)+'\n');print({k:v['median_visible_work_fraction'] for k,v in reports.items()})
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Report bounded diagnostic costs; preserve ambiguous graph attribution."""
import argparse,json,statistics
from collections import defaultdict
from pathlib import Path
def category(name):
    if 'HadInKernel' in name:return 'EXL3_HadIn'
    if 'HadOutKernel' in name:return 'EXL3_HadOut'
    if 'DpasKernel' in name:return 'EXL3_GEMM'
    if 'ReduceSplitK' in name:return 'M04_or_generic_splitK_reduction'
    if 'FMHA' in name:return 'M04_or_generic_attention_mainloop'
    if 'fused_qk' in name:return 'QKV_norm_RoPE'
    if 'layer_norm' in name or 'rms_norm' in name:return 'norm_requires_shape_route_attribution'
    if 'gdn::' in name or 'gated_delta' in name or 'causal_conv1d' in name:return 'GDN_core_and_state'
    if 'copy' in name.lower() or 'memcpy' in name.lower():return 'copy_requires_route_attribution'
    return 'other'
def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();reports={}
    for events in a.root.glob('*-events.json'):
        name=events.name.removesuffix('-events.json');d=json.loads(events.read_text());cycles=[c for c in d['cycles'] if c.get('batch') and not c['batch']['has_prefill']]
        stages=defaultdict(list)
        for c in cycles:
            for s in c['stages']:stages[s['stage']].append(s['gpu_timeline_ms'])
        trace=json.loads((a.root/(name+'-trace.json')).read_text())['traceEvents']
        annotations=[e for e in trace if e.get('cat')=='gpu_user_annotation' and e.get('name','').startswith('exl3_diagnostic/')]
        kernels=[e for e in trace if e.get('cat')=='kernel'];counts=defaultdict(lambda:dict(kernel_count=0,kernel_ms=0.))
        detailed=defaultdict(lambda:dict(kernel_count=0,kernel_ms=0.))
        for e in kernels:
            labels={v['name'].rsplit('/',1)[-1] for v in annotations if v['tid']==e['tid'] and v['ts']-.1<=e['ts']<=v['ts']+v['dur']+.1}
            stage=next(iter(labels)) if len(labels)==1 else 'ambiguous' if labels else 'unattributed'
            for v in [counts[(stage,category(e['name']))],detailed[(stage,e['name'])]]:v['kernel_count']+=1;v['kernel_ms']+=e['dur']/1000
        reports[name]=dict(decode_cycles=len(cycles),median_round_span_ms=statistics.median(c['cycle_gpu_timeline_ms'] for c in cycles),
            median_event_spans_ms={k:statistics.median(v) for k,v in stages.items()},
            graph_batch_forms=[dict(b) for b in {json.dumps(c['batch'],sort_keys=True):c['batch'] for c in cycles}.values()],
            kernel_groups=[dict(stage=k[0],category=k[1],**v) for k,v in sorted(counts.items(),key=lambda x:-x[1]['kernel_ms'])],
            kernels=[dict(stage=k[0],name=k[1],**v) for k,v in sorted(detailed.items(),key=lambda x:-x[1]['kernel_ms'])],
            limitations=['Kernel-duration sums are not nonoverlapping critical-path fractions.', 'Event spans include enqueue gaps; medians are not additive.', 'Names alone do not prove exact operator shapes or distinguish every target/MTP norm.', 'Trace start/stop perturb the diagnostic wave; control timings are separate.'])
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(status='P0_DIAGNOSTIC_COST_REPORT_NOT_SPEEDUP',reports=reports),indent=2)+'\n')
    print(json.dumps({k:{f:v for f,v in r.items() if f not in ['kernels','graph_batch_forms','kernel_groups']} for k,r in reports.items()},indent=2))
if __name__=='__main__':main()

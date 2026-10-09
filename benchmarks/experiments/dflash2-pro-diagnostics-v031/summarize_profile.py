#!/usr/bin/env python3
"""Summarize event spans and disjoint kernel groups, never serving throughput."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def kernel_category(stage, name):
    if 'target_body' in stage:
        if 'DpasKernel' in name or 'HadInKernel' in name or 'HadOutKernel' in name:
            return 'target_exl3_linear'
        if 'gdn::' in name or 'gated_delta' in name or 'causal_conv1d_spec' in name:
            return 'target_gdn_and_speculative_state'
        if 'FMHA' in name or 'shared_kv_verify' in name or 'ReduceSplitK' in name:
            return 'target_attention_M04_and_reduction'
        return 'target_body_other'
    if 'target_head' in stage:
        return 'target_full_head'
    if 'verification_sampler' in stage:
        return 'target_sampling_and_rejection'
    if 'draft_proposal' in stage:
        if 'DpasKernel<6' in name or 'HadInKernel' in name or 'HadOutKernel' in name or 'Sbtopk' in name:
            return 'draft_shared_head_and_top16'
        if 'FMHA' in name or 'ReduceSplitK' in name:
            return 'draft_own_attention'
        if 'grouped_conv' in name or 'selector' in name or 'einsum' in name:
            return 'draft_selector_and_convolution'
        if 'gemm_kernel' in name:
            return 'draft_dense_projections_unresolved_FC_KV_backbone_selector'
        if 'reshape_and_cache' in name or 'rotary' in name or 'rope' in name:
            return 'draft_RoPE_and_cache_write'
        return 'draft_other'
    return 'commit_and_other'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('worker', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads((args.worker/'event-profile.json').read_text())
    cycles = [c for c in profile['cycles'] if c.get('batch') and not c['batch']['has_prefill']]
    stages = defaultdict(list)
    for cycle in cycles:
        for stage in cycle['stages']:
            stages[stage['stage']].append(stage['gpu_timeline_ms'])
    medians = {name: statistics.median(values) for name, values in stages.items()}
    timeline = statistics.median(c['cycle_gpu_timeline_ms'] for c in cycles)
    events = json.loads((args.worker/'kernel-trace.json').read_text())['traceEvents']
    annotations = [e for e in events if e.get('cat') == 'gpu_user_annotation' and e['name'].startswith('exl3_diagnostic/')]
    kernels = [e for e in events if e.get('cat') == 'kernel']
    buckets = defaultdict(lambda: dict(kernel_count=0, total_gpu_kernel_ms=0.))
    detailed = defaultdict(lambda: dict(kernel_count=0, total_gpu_kernel_ms=0.))
    for e in kernels:
        matches = [a for a in annotations if a['tid']==e['tid'] and a['ts']-.1 <= e['ts'] <= a['ts']+a['dur']+.1]
        # Kineto graph annotation envelopes can overlap the next runner cycle.
        # Label ambiguous attribution instead of double-counting or guessing.
        labels = {a['name'].rsplit('/', 1)[-1] for a in matches}
        stage = next(iter(labels)) if len(labels)==1 else 'ambiguous_graph_annotation' if labels else 'unattributed'
        category = kernel_category(stage, e['name'])
        for bucket in (buckets[category], detailed[(stage, e['name'])]):
            bucket['kernel_count'] += 1
            bucket['total_gpu_kernel_ms'] += e['dur']/1000
    draft = medians['draft_proposal_including_head_and_sampler']
    result = dict(status='DIAGNOSTIC_COST_SPLIT_NOT_THROUGHPUT', decode_cycles=len(cycles),
        median_round_event_span_ms=timeline, median_stage_event_spans_ms=medians,
        complete_draft_elimination_upper_bound_fraction=draft/timeline,
        complete_draft_elimination_ideal_speedup=timeline/(timeline-draft),
        kernel_buckets=dict(buckets),
        detailed_kernels=[dict(stage=s, name=n, **v) for (s,n),v in sorted(detailed.items(), key=lambda x:-x[1]['total_gpu_kernel_ms'])],
        limits=['Event spans include dispatch gaps and medians are not additive',
                'Profiler wave is perturbed and separate from tensor capture',
                'Dense gemm symbols do not uniquely separate feature-FC, context-KV, backbone and selector projections; do not invent component attribution',
                'Overlapping GPU streams can make sum of kernel durations exceed wall time'])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('detailed_kernels','kernel_buckets')}, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Summarize coarse event spans without claiming visibility inside XPU graphs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def receipt(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def stats(values):
    return {'count': len(values), 'mean': statistics.mean(values), 'min': min(values), 'max': max(values)}


def summarize_events(path):
    data = json.loads(path.read_text())
    groups = defaultdict(list)
    without_body = 0
    for cycle in data['cycles']:
        batch = cycle['batch']
        if batch is None:
            without_body += 1
            continue
        key = (batch['has_prefill'], batch['requests'], batch['actual_rows'], batch['padded_rows'],
               batch['target_graph_mode'], batch['target_graph_bucket_rows'])
        groups[key].append(cycle)
    rows = []
    for key, cycles in groups.items():
        stages = defaultdict(list)
        for cycle in cycles:
            for stage in cycle['stages']:
                stages[stage['stage']].append(stage['gpu_timeline_ms'])
        rows.append({'has_prefill': key[0], 'requests': key[1], 'actual_rows': key[2],
                     'padded_rows': key[3], 'target_graph_mode': key[4], 'graph_bucket_rows': key[5],
                     'cycles': len(cycles), 'cycle_gpu_timeline_ms': stats([c['cycle_gpu_timeline_ms'] for c in cycles]),
                     'stage_gpu_timeline_ms': {name: stats(values) for name, values in stages.items()}})
    assert rows
    return {'evidence': receipt(path), 'cycles': len(data['cycles']),
            'cycles_without_target_body': without_body, 'groups': rows,
            'scope': data['scope'], 'draft_scope': data['draft_scope'],
            'missing_fine_components': data['missing_fine_components']}


def summarize_trace(path):
    data = json.loads(path.read_text()); meta_path = Path(str(path) + '.meta.json')
    meta = json.loads(meta_path.read_text())
    assert meta['status'] == 'COMPLETE' and meta['runner_cycles'] == len(meta['batches']) == 8
    kernels = [e for e in data['traceEvents'] if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    totals = defaultdict(lambda: [0.0, 0])
    for event in kernels:
        totals[event['name']][0] += event.get('dur', 0)
        totals[event['name']][1] += 1
    top = sorted(totals.items(), key=lambda item: -item[1][0])[:20]
    return {'evidence': receipt(path), 'metadata_evidence': receipt(meta_path), 'metadata': meta,
            'observed_kernel_events': len(kernels),
            'top_observed_kernels': [{'name': name, 'sum_duration_ms': values[0] / 1000, 'count': values[1]}
                                    for name, values in top],
            'attention_named_events': sum(any(word in e['name'].lower() for word in ['attention', 'attn', 'fmha'])
                                         for e in kernels),
            'coverage_note': 'Name matches are a heuristic. FULL graph body and draft kernels are not demonstrated as separately visible by these traces. Observing no attention-named kernel does not prove attention is cheap. Require a bounded eager trace or another profiler with proven graph-node coverage before M04 materiality conclusions.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); root = args.campaign.resolve()
    state = json.loads((root / 'campaign.json').read_text())
    assert state['status'] == 'COMPLETE' and state['compact']
    assert [a['depth'] for a in state['arms']] == [3, 4]
    assert all(a['status'] == 'COMPLETE' for a in state['arms'])
    assert not args.out.exists(), 'Fresh assessment required'
    arms = []
    for arm in state['arms']:
        directory = root / f'mtp{arm["depth"]}'
        summary_path = directory / 'summary.json'; summary = json.loads(summary_path.read_text())
        assert receipt(summary_path)['sha256'] == arm['summary_sha256']
        assert summary['status'] == 'COMPLETE' and summary['compact']
        assert {(w['name'], w['generated_tokens']) for w in summary['waves']} == \
               {('code-4096-c1', 128), ('code-49152-c4', 512), ('code-102752-c1', 128)}
        assert summary['panel_sha256'] == state['panel_sha256']
        arms.append({'depth': arm['depth'], 'summary_evidence': receipt(summary_path),
                     'engine_config': summary['engine_config'], 'head_inventory': summary['vocabulary_inventory'],
                     'event_waves': [{**wave, 'event_summary': summarize_events(directory / (wave['name'] + '.json'))}
                                     for wave in summary['waves']],
                     'kernel_traces': [{**trace, 'trace_summary': summarize_trace(directory / (trace['name'] + '.json'))}
                                       for trace in summary['kernel_traces']]})
    result = {'status': 'COMPLETE_COARSE_PROFILE_GRAPH_KERNEL_DECOMPOSITION_UNRESOLVED',
              'campaign_evidence': receipt(root / 'campaign.json'), 'image_id': state['image_id'],
              'source_sha256': state['source_sha256'], 'arms': arms,
              'scope': 'Instrumented event timeline spans including dispatch gaps, not instrument-free throughput. Group by actual active requests/rows; nominal C4 does not establish four simultaneous pure-decode requests.',
              'remaining': ['Fine target attention and fused draft body/head/sampler decomposition',
                            'Generated-path margin/correctness check', 'Depth decision and full-vocabulary ablation']}
    args.out.mkdir(parents=True)
    (args.out / 'assessment.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'out': str(args.out)}))


if __name__ == '__main__':
    main()

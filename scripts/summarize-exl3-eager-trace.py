#!/usr/bin/env python3
"""Attribute eager XPU kernels using correlation IDs and CPU stage scopes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--serving', type=Path, required=True)
    args = parser.parse_args(); root = args.campaign.resolve()
    state = json.loads((root / 'campaign.json').read_text())
    summary_path = root / 'capture/summary.json'; summary = json.loads(summary_path.read_text())
    assert state['status'] == summary['status'] == 'COMPLETE'
    assert sha(summary_path) == state['summary_sha256'] and summary['engine_config']['enforce_eager']
    path = root / 'capture/kernel-trace.json'; events = json.loads(path.read_text())['traceEvents']
    meta_path = Path(str(path) + '.meta.json'); meta = json.loads(meta_path.read_text())
    assert meta['pure_decode_cycles'] == meta['runner_cycles'] == 8
    assert all(b['requests'] == 1 and b['actual_rows'] == 4 and b['target_graph_mode'].endswith('NONE')
               for b in meta['batches'])
    ops = {e['args']['External id']: e for e in events if e.get('cat') == 'cpu_op'
           and 'External id' in e.get('args', {})}
    ranges = [e for e in events if e.get('cat') == 'user_annotation'
              and e.get('name', '').startswith('exl3_diagnostic/')]
    links = {}
    for event in events:
        if event.get('cat') in ['xpu_driver', 'xpu_runtime'] and 'correlation' in event.get('args', {}):
            links[event['args']['correlation']] = event['args'].get('External id')
    totals = defaultdict(lambda: {'kernel_events': 0, 'kernel_ms': 0.0,
                                 'attention_main_events': 0, 'attention_reduce_events': 0, 'attention_ms': 0.0})
    kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    for event in kernels:
        op = ops.get(links.get(event.get('args', {}).get('correlation')))
        parent = next((r for r in ranges if op and r['tid'] == op['tid'] and r['ts'] <= op['ts']
                       and op['ts'] + op['dur'] <= r['ts'] + r['dur'] + 1), None)
        stage = parent['name'].split('/')[-1] if parent else 'unmapped'
        row = totals[stage]; row['kernel_events'] += 1; row['kernel_ms'] += event['dur'] / 1000
        is_reduce = 'XeReduceSplitKTileScheduler' in event['name']
        is_main = 'XeFMHAFwdSplitKVKernel' in event['name'] and not is_reduce
        if is_main or is_reduce:
            assert parent is not None, 'Attention kernel lacks a proven CPU stage parent'
            row['attention_ms'] += event['dur'] / 1000
            row['attention_main_events' if is_main else 'attention_reduce_events'] += 1
    assert totals['target_body']['attention_main_events'] == totals['target_body']['attention_reduce_events'] == 128
    draft = totals['draft_proposal_including_head_and_sampler']
    assert draft['attention_main_events'] == draft['attention_reduce_events'] == 24
    for row in totals.values():
        row['kernel_ms_per_cycle'] = row['kernel_ms'] / 8
        row['attention_ms_per_cycle'] = row['attention_ms'] / 8
        row['attention_share_of_summed_kernel_ms'] = row['attention_ms'] / row['kernel_ms']
    serving = json.loads((args.serving / 'campaign.json').read_text())
    assert serving['status'] == 'COMPLETE' and serving['image_id'] == state['image_id']
    assert serving['panel_sha256'] == state['panel_sha256']
    wave = next(w for w in serving['arms'][0]['waves'] if
                (w['window'], w['concurrency'], w['cache_mode']) == ('code-102752', 1, 'cold'))
    raw_result = Path(wave['reused_result']); assert sha(raw_result) == wave['reused_result_sha256']
    tokens = json.loads(raw_result.read_text())['responses'][0]['token_ids'][:128]
    graph_hash = hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()
    result = {'status': 'COMPLETE_EAGER_ATTENTION_MATERIALITY_NOT_SERVING_SPEEDUP',
              'image_id': state['image_id'], 'engine_config': summary['engine_config'],
              'metadata': meta, 'kernel_events': len(kernels), 'stage_kernel_sums': dict(totals),
              'streams': dict(Counter(str(e['tid']) for e in kernels)),
              'graph_first128_output_match': graph_hash == summary['output_ids_sha256'],
              'evidence': {str(p.resolve()): sha(p) for p in
                           [root / 'campaign.json', summary_path, path, meta_path, raw_result]},
              'interpretation': 'Actual128 target attention main+128 reduction kernels and24+24 draft attention kernels are correlated to8 eager cycles. Summed kernel durations are not wall-clock or serving-graph timings. Attention is material enough for a guarded M04 prototype; any gain requires numerical and actual serving-graph validation. One code128-token match does not close generated quality.'}
    assert not args.out.exists(); args.out.mkdir(parents=True)
    (args.out / 'assessment.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'target_attention_share': totals['target_body']['attention_share_of_summed_kernel_ms'],
                      'target_attention_ms_per_cycle': totals['target_body']['attention_ms_per_cycle'],
                      'graph_first128_output_match': result['graph_first128_output_match']}))


if __name__ == '__main__':
    main()

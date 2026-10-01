#!/usr/bin/env python3
"""Audit a completed compact MTP screen and preserve per-case comparisons."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_wave(root, index, arm, wave):
    if 'reused_result' in wave:
        path = Path(wave['reused_result'])
        assert sha(path) == wave['reused_result_sha256'], 'Reused raw result changed'
    else:
        path = root / f'arm-{index}-mtp{arm["depth"]}' / f'{wave["window"]}-c{wave["concurrency"]}-{wave["cache_mode"]}' / 'result.json'
    raw = json.loads(path.read_text())
    summary = {k: v for k, v in raw.items() if k != 'responses'}
    assert summary == {k: v for k, v in wave.items() if not k.startswith('reused_result')}
    c = wave['concurrency']
    assert raw['native']['completed'] == c and len(raw['responses']) == c
    assert raw['native']['generation_tokens'] == c * 512
    for response in raw['responses']:
        assert response['prompt_ids_verified'] and len(response['token_ids']) == 512
        assert response['usage']['prompt_tokens'] == wave['context_tokens']
        digest = hashlib.sha256(json.dumps(response['token_ids'], separators=(',', ':')).encode()).hexdigest()
        assert digest == response['output_ids_sha256']
    return raw, {'path': str(path.resolve()), 'sha256': sha(path)}


def change(a, b):
    return (b / a - 1) * 100 if a else None


def paired_row(a, b):
    assert (a['window'], a['concurrency'], a['cache_mode']) == (b['window'], b['concurrency'], b['cache_mode'])
    first_differences = [next((i for i, (x, y) in enumerate(zip(r3['token_ids'], r4['token_ids'])) if x != y), None)
                         for r3, r4 in zip(a['responses'], b['responses'])]
    keys = ['native_request_weighted_decode_tps', 'client_aggregate_decode_tps',
            'native_prefill_tps', 'wave_aggregate_output_tps', 'generated_tokens_per_round',
            'joules_per_output_token', 'wall_s']
    row = {k: a[k] for k in ['window', 'domain', 'context_tokens', 'concurrency', 'cache_mode']}
    row['metrics'] = {k: {'mtp3': a[k], 'mtp4': b[k],
                         'mtp4_change_pct': change(a[k], b[k]) if a[k] is not None and b[k] is not None else None}
                      for k in keys}
    row['native_accounting'] = {f'mtp{depth}': {k: value['native'][k] for k in
        ['cached_tokens', 'prefill_tokens', 'preemptions', 'prefill_seconds', 'decode_seconds',
         'accepted_tokens', 'draft_tokens']} for depth, value in [(3, a), (4, b)]}
    row['client_ttft_s'] = {'mtp3': a['client_ttft_s'], 'mtp4': b['client_ttft_s']}
    row['first_output_difference_by_request'] = first_differences
    row['exact_output_matches'] = sum(x is None for x in first_differences)
    row['cache_hit_fraction'] = {f'mtp{depth}': value['native']['cached_tokens'] / value['native']['prompt_tokens']
                               for depth, value in [(3, a), (4, b)]}
    row['interpretation'] = (
        'Observed serving performance, not isolated kernel speed. Cold C4 may mix decode with other prefills. '
        'Warm cache residency can differ between depths; report actual hits and misses. '
        'Output sequences can differ, changing acceptance work. No single-pass confidence interval.')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Fresh assessment required'
    root = args.campaign.resolve()
    state = json.loads((root / 'campaign.json').read_text())
    assert state['status'] == 'COMPLETE' and state['scope'] == 'SERVING_COMPACT_SCREEN'
    assert state['order'] == [3, 4] and state['planned_waves'] == 36
    assert [a['depth'] for a in state['arms']] == [3, 4]
    a3, a4 = state['arms']
    assert a3['settings']['speculative_config'] == {'method': 'mtp', 'num_speculative_tokens': 3}
    assert a4['settings']['speculative_config'] == {'method': 'mtp', 'num_speculative_tokens': 4}
    assert {k: v for k, v in a3['settings'].items() if k != 'speculative_config'} == \
           {k: v for k, v in a4['settings'].items() if k != 'speculative_config'}
    arms, evidence = [], []
    for index, arm in enumerate(state['arms']):
        assert len(arm['waves']) == 18
        values = {}
        for wave in arm['waves']:
            key = (wave['window'], wave['concurrency'], wave['cache_mode'])
            assert key not in values
            values[key], receipt = load_wave(root, index, arm, wave)
            evidence.append(receipt)
        arms.append(values)
    assert arms[0].keys() == arms[1].keys()
    wanted = {(f'{domain}-{ctx}', c, cache) for domain in ['code', 'prose']
              for ctx in [4096, 49152] for c in [1, 4] for cache in ['cold', 'warm']}
    wanted |= {('code-102752', 1, cache) for cache in ['cold', 'warm']}
    assert set(arms[0]) == wanted, 'Unexpected compact case matrix'
    rows = [paired_row(value, arms[1][key]) for key, value in arms[0].items()]
    requests = sum(row['concurrency'] for row in rows) * 2
    assert requests == 84
    assessment = {'status': 'COMPLETE_SCREEN_NOT_RELEASE_QUALIFICATION',
        'campaign_sha256': sha(root / 'campaign.json'), 'image_id': state['image_id'],
        'panel_sha256': state['panel_sha256'], 'waves': 36, 'requests': requests,
        'comparison_design': state['comparison_design'], 'paired_cases': rows, 'raw_results': evidence,
        'selection': 'No universal depth selected from single-pass screening. Use component/row evidence and targeted repeats for ambiguous cells; generated-output correctness remains a release gate.'}
    args.out.mkdir(parents=True)
    (args.out / 'assessment.json').write_text(json.dumps(assessment, indent=2) + '\n')
    lines = ['# Compact MTP3/MTP4 serving screen', '',
        'One historical MTP3 pass and one fresh MTP4 pass, 36 waves / 84 requests. '
        'This is screening, not a completed ABBA or final release qualification.', '',
        'C1 uses native request-weighted decode; C4 uses client aggregate decode over the union of observed request decode intervals. '
        'Both are serving observations, including scheduling/prefill interference. Actual cache hits are shown because a warm label does not establish identical residency.', '',
        '| Context / workload | C | Cache | MTP3 decode tok/s | MTP4 decode tok/s | Change | Cache hits 3 / 4 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for row in rows:
        key = 'native_request_weighted_decode_tps' if row['concurrency'] == 1 else 'client_aggregate_decode_tps'
        m = row['metrics'][key]; hit = row['cache_hit_fraction']
        lines.append(f'| {row["window"]} | {row["concurrency"]} | {row["cache_mode"]} | {m["mtp3"]:.2f} | {m["mtp4"]:.2f} | {m["mtp4_change_pct"]:+.1f}% | {hit["mtp3"]:.1%} / {hit["mtp4"]:.1%} |')
    lines += ['', assessment['selection'], '',
        'Exact output hashes and first differing token positions are retained in assessment.json. '
        'Sequence variation is not assumed to be harmless numerical rounding. Instrumented component traces remain separate from these throughput observations.', '']
    (args.out / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'status': assessment['status'], 'waves': 36, 'requests': requests, 'out': str(args.out)}))


if __name__ == '__main__':
    main()

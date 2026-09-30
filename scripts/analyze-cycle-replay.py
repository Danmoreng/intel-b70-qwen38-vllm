#!/usr/bin/env python3
"""Compare complete cycles only when frozen inputs and restored states agree."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    groups = {}
    for path in sorted(args.root.glob('arm-*/cycle-replay-c*.json')):
        data = json.loads(path.read_text())
        if not data['repeatability_passed'] or not data['registered_initial_state_bytes_verified_each_repeat']:
            raise RuntimeError(f'unverified starting state: {path}')
        if not data.get('head_and_sampler_inputs_frozen') or not data.get('real_sampler_commit_matches_reference_each_repeat'):
            raise RuntimeError(f'head/sampler work is not verified as frozen: {path}')
        rows = data['rows']
        if len(rows) < 15:
            raise RuntimeError(f'insufficient repeated cycles: {path}')
        signatures = {key: {row[key] for row in rows}
                      for key in ('input_signature', 'draft_signature')}
        if any(len(values) != 1 for values in signatures.values()):
            raise RuntimeError(f'varying frozen work: {path}')
        padded_rows = {len(row['prepared_inputs']['is_padding']) for row in rows}
        if padded_rows != {data['actual_query_rows']} or any(any(row['prepared_inputs']['is_padding']) for row in rows):
            raise RuntimeError(f'unexpected padding in full-cycle replay: {path}')
        identity = (data['fixture_sha256'], data['actual_query_rows'],
                    json.dumps(data['fixed_commit'], sort_keys=True),
                    *(next(iter(signatures[key])) for key in signatures))
        groups.setdefault(data['request_count'], []).append((path, data, identity))
    results = []
    for concurrency, entries in sorted(groups.items()):
        if len({identity for _, _, identity in entries}) != 1:
            raise RuntimeError(f'arms used different work: C{concurrency}')
        arms = {}
        for path, data, _ in entries:
            arm = path.parent.name.split('-', 2)[2]
            rows = data['rows'][3:]  # discard the first three replay warmups
            arms[arm] = {
                'source': str(path.resolve()), 'timed_repeats': len(rows),
                'wall_median_ms': statistics.median(row['wall_ms'] for row in rows),
                'stream_median_ms': statistics.median(row['stream_ms'] for row in rows),
                'wall_range_ms': [min(row['wall_ms'] for row in rows), max(row['wall_ms'] for row in rows)],
                'target_bitwise_repeatable': data['target_bitwise_repeatable'],
                'target_max_abs_reference': max(row['target_max_abs_reference'] for row in data['rows']),
                'registered_initial_state_bytes_verified_each_repeat': True,
            }
        comparison = {}
        if 'current' in arms and 'fused' in arms:
            for mode in ('wall', 'stream'):
                key = mode + '_median_ms'
                comparison[mode + '_cycle_reduction_pct'] = 100 * (1 - arms['fused'][key] / arms['current'][key])
        results.append({'concurrency': concurrency, 'actual_query_rows': entries[0][1]['actual_query_rows'],
                        'padded_query_rows': entries[0][1]['actual_query_rows'],
                        'fixture_sha256': entries[0][1]['fixture_sha256'], 'arms': arms, **comparison})
    if not results:
        raise RuntimeError('no complete replay results')
    result = {'scope': 'complete target/head/sampler/commit/MTP4 draft cycle; state restore outside timing',
              'numeric_gate_note': 'Identical restored registered state and frozen work are verified. Target output differences remain reported and require independent numeric and quality gates.',
              'results': results}
    (args.root / 'cycle-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

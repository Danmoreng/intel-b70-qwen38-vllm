#!/usr/bin/env python3
"""Compare cold prefill/mixed fixtures and each frozen quality task's result."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--quality-reference', type=Path, required=True)
    args = parser.parse_args()
    images = json.loads((args.root / 'provenance.json').read_text())['images']
    phases = {path.name.split('-', 2)[2]: path for path in args.root.glob('arm-*')}
    prefill = []
    for scenario in ('phase-32k-c1', 'phase-128k-c1'):
        rows = {}
        for arm in ('current', 'fused'):
            phase = phases[arm]
            folder = phase / f'{scenario}-r1'
            data = json.loads((folder / 'summary.json').read_text())
            if any(data[key] for key in ('prompt_tokens_cached', 'preemptions', 'prefill_recompute_excess')):
                raise RuntimeError(f'non-cold or recomputed prefill: {folder}')
            for filename in ('prompt-1.txt', 'request-1.json'):
                if (folder / filename).read_bytes() != (phases['current'] / f'{scenario}-r1' / filename).read_bytes():
                    raise RuntimeError(f'prefill fixtures differ: {folder}')
            warm = json.loads((phase / 'warmups' / f'{scenario}-r0/summary.json').read_text())
            rows[arm] = {'prefill_tokens_per_s': data['native_prefill_compute_tokens_per_s'],
                         'ttft_s': data['requests'][0]['ttft_s'],
                         'warmup_prefill_tokens_per_s': warm['native_prefill_compute_tokens_per_s'],
                         'round_equivalent_ms': data['fully_overlapped_round_equivalent_ms'],
                         'source': str((folder / 'summary.json').resolve())}
        prefill.append({'scenario': scenario, 'arms': rows,
                        'candidate_prefill_rate_change_pct': 100 * (rows['fused']['prefill_tokens_per_s'] / rows['current']['prefill_tokens_per_s'] - 1)})
    mixed = {arm: json.loads((phases[arm] / 'mixed.json').read_text()) for arm in ('current', 'fused')}
    for key in ('task_id', 'task_context_sha256', 'task_prompt_sha256', 'background_prompt_sha256', 'namespace'):
        if mixed['current'][key] != mixed['fused'][key]:
            raise RuntimeError(f'mixed fixture differs: {key}')
    for arm, row in mixed.items():
        if row['image_id'] != images[arm]:
            raise RuntimeError(f'mixed image identity differs: {arm}')
        if row['background_forced_token_id'] is not None or any(row['metrics'][key] for key in ('cached_tokens', 'preemptions')):
            raise RuntimeError(f'invalid mixed control: {arm}')
    quality = []
    for task_set in ('32k', '128k'):
        def load(path):
            return {row['id']: row for row in map(json.loads, path.read_text().splitlines())}
        reference = load(args.quality_reference / f'{task_set}.jsonl')
        candidate = load(phases['fused'] / f'quality-{task_set}.jsonl')
        if set(reference) != set(candidate):
            raise RuntimeError(f'quality tasks missing or changed: {task_set}')
        for task_id, row in candidate.items():
            if row['image'] != images['fused'] or reference[task_id]['image'] != images['current']:
                raise RuntimeError(f'quality image identity differs: {task_id}')
            for key in ('context_sha256', 'prompt_sha256'):
                if row[key] != reference[task_id][key]:
                    raise RuntimeError(f'quality fixture differs: {task_id}/{key}')
        passed = lambda row: row['score']['passed']
        quality.append({'task_set': task_set, 'count': len(candidate),
                        'reference_passed': sum(map(passed, reference.values())),
                        'candidate_passed': sum(map(passed, candidate.values())),
                        'new_failures': [task_id for task_id in candidate if passed(reference[task_id]) and not passed(candidate[task_id])],
                        'candidate_failures': [task_id for task_id, row in candidate.items() if not passed(row)],
                        'source': str((phases['fused'] / f'quality-{task_set}.jsonl').resolve())})
    result = {'prefill': prefill,
              'mixed': {arm: {
                  **{key: row[key] for key in ('image_id', 'task_score', 'task_timing', 'metrics')},
                  'background': {key: value for key, value in row['background'].items() if key != 'stream_piece_times_s'},
              } for arm, row in mixed.items()},
              'quality': quality,
              'quality_passed_without_new_failures': all(not row['new_failures'] for row in quality),
              'mixed_task_passed': all(row['task_score']['passed'] for row in mixed.values())}
    (args.root / 'validation-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

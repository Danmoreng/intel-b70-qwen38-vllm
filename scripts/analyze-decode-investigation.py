#!/usr/bin/env python3
"""Recompute paired decode windows from raw metrics and verify request identity."""
import argparse
import hashlib
import json
from pathlib import Path

from decode_overlap import pool_overlap, summarize_overlap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    groups = {}
    identities = {}
    waves = []
    for phase in sorted(args.root.glob('arm-*')):
        arm = phase.name.split('-', 2)[2]
        for path in sorted(phase.glob('*/summary.json')):
            summary = json.loads(path.read_text())
            folder = path.parent
            requests = summary['requests']
            if len(requests) != summary['scenario']['concurrency'] or any(
                    row['usage']['completion_tokens'] != summary['scenario']['output_tokens'] for row in requests):
                raise RuntimeError(f'output work differs from frozen budget: {path}')
            samples = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
            overlap = summarize_overlap(samples, max(row['ttft_s'] for row in requests),
                min(row['wall_s'] for row in requests), summary['scenario']['concurrency'])
            if not overlap['fully_overlapped_counter_window_valid']:
                raise RuntimeError(f'invalid overlap {path}: {overlap}')
            first, last = overlap['fully_overlapped_sample_start_s'], overlap['fully_overlapped_sample_end_s']
            summary.update(overlap)
            summary['fully_overlapped_decode_sampled_s'] = last - first
            window = [s for s in samples if 'error' not in s and first <= s['elapsed_s'] <= last]
            generated = window[-1]['generation_tokens'] - window[0]['generation_tokens']
            summary['fully_overlapped_generation_tokens'] = generated
            groups.setdefault((summary['scenario']['name'], arm), []).append(summary)
            for index in range(summary['scenario']['concurrency']):
                for kind, suffix in (('request', 'json'), ('prompt', 'txt')):
                    source = folder / f'{kind}-{index + 1}.{suffix}'
                    identity = (folder.name, source.name)
                    digest = hashlib.sha256(source.read_bytes()).hexdigest()
                    identities.setdefault(identity, {})[arm] = digest
            waves.append({'arm': arm, 'case': folder.name,
                          'round_equivalent_ms': overlap['fully_overlapped_round_equivalent_ms'],
                          'acceptance': overlap['fully_overlapped_speculative_acceptance'],
                          'output_tokens_per_s': generated / (last - first),
                          'sampled_seconds': last - first,
                          'preemptions': summary['preemptions'],
                          'cached_tokens': summary['prompt_tokens_cached'],
                          'recomputed_tokens': summary['prefill_recompute_excess'],
                          'completion_tokens': sum(row['usage']['completion_tokens'] for row in requests),
                          'all_requested_output_tokens_emitted': True})
    pooled = []
    for (scenario, arm), rows in sorted(groups.items()):
        summary = pool_overlap(rows)
        summary.update(scenario=scenario, arm=arm, repeats=len(rows),
            output_tokens_per_s=sum(row['fully_overlapped_generation_tokens'] for row in rows)
                / sum(row['fully_overlapped_decode_sampled_s'] for row in rows))
        pooled.append(summary)
    mismatches = [key for key, arms in identities.items() if len(set(arms.values())) != 1]
    if mismatches:
        raise RuntimeError(f'paired request files differ: {mismatches}')
    paired = sum(len(arms) > 1 for arms in identities.values())
    output = {'waves': waves, 'pooled': pooled,
              'byte_identical_paired_files': paired,
              'identity_note': 'request metadata and exact prompt bytes, not output identity',
              'identity_sha256': {f'{case}/{name}': arms for (case, name), arms in sorted(identities.items())}}
    (args.root / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'pooled': pooled, 'byte_identical_paired_files': paired}, indent=2))


if __name__ == '__main__':
    main()

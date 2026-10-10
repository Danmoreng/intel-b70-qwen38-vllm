#!/usr/bin/env python3
"""Observed comparability for named controls, including missing early coverage."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--telemetry', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    campaign = json.loads((args.root / 'campaign.json').read_text())
    assert campaign['status'] == 'COMPLETE_NAMED_FROZEN_REFERENCES_REVIEW_REQUIRED'
    samples = [json.loads(line) for line in args.telemetry.read_text().splitlines()]
    assert samples
    caps = [int(value) for row in samples for key, value in row['values'].items()
            if key.endswith('/power1_cap')]
    assert len(caps) == len(samples) and set(caps) == {180000000}
    arms = []
    for block in campaign['blocks']:
        for role in block['order']:
            arm = block['arms'][role]
            initial = args.root / f"block-{block['index']}-{role}/inventory-initial.json"
            result = args.root / arm['results_path']
            start, end = initial.stat().st_mtime, result.stat().st_mtime
            selected = [row for row in samples if start <= row['unix'] <= end]
            metrics = {}
            for key in sorted({key for row in selected for key in row['values']}):
                values = [row['values'][key] for row in selected if key in row['values']]
                try:
                    numeric = [float(value) for value in values]
                except ValueError:
                    metrics[key] = {'observed_values': sorted(set(values))}
                else:
                    metrics[key] = {'min': min(numeric), 'median': statistics.median(numeric),
                                    'max': max(numeric), 'nonzero_samples': sum(value != 0 for value in numeric)}
            arms.append({'block': block['index'], 'role': role, 'sampled_rows': len(selected),
                         'interval_unix': [start, end], 'early_interval_unobserved': samples[0]['unix'] > start,
                         'metrics': metrics})
    report = {'status': 'COMPLETE_OBSERVED_COMPARABILITY_PARTIAL_FIRST_ARM_COVERAGE',
              'power_cap_w': 180, 'samples': len(samples), 'first_unix': samples[0]['unix'],
              'last_unix': samples[-1]['unix'], 'arms': arms,
              'campaign_sha256': hashlib.sha256((args.root / 'campaign.json').read_bytes()).hexdigest(),
              'telemetry_sha256': hashlib.sha256(args.telemetry.read_bytes()).hexdigest(),
              'limits': 'Read-only 15-second sysfs snapshots, not continuous monitoring. This monitor began during the first baseline arm; earlier temperatures/clocks are unobserved and cannot be reconstructed. Per-arm intervals use host inventory/result file-write times outside measurements and include warmups. Power-limit throttling can occur under the selected 180-W cap; thermal flags are reported separately. No energy benchmark or power-efficiency comparison was repeated.'}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'samples': len(samples), 'arm_sample_counts': [(arm['block'], arm['role'], arm['sampled_rows']) for arm in arms]}))


if __name__ == '__main__':
    main()

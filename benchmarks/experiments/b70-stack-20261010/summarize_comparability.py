"""Summarize observed clocks/thermal flags without an energy benchmark."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    campaigns = []
    for root in args.root:
        path = root / 'comparability.jsonl'
        samples = [json.loads(line) for line in path.read_text().splitlines()]
        assert samples
        caps = [int(value) for row in samples for key, value in row['values'].items()
                if key.endswith('/power1_cap')]
        assert len(caps) == len(samples) and set(caps) == {180000000}
        record = dict(root=str(root), samples=len(samples), first_unix=samples[0]['unix'],
                      last_unix=samples[-1]['unix'], power_cap_w=180,
                      source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), arms=[])
        for initial in sorted(root.glob('block-*/inventory-initial.json')):
            final = initial.with_name('inventory-final.json')
            if not final.exists():
                continue
            start, end = initial.stat().st_mtime, final.stat().st_mtime
            selected = [row for row in samples if start <= row['unix'] <= end]
            metrics = {}
            for key in sorted({key for row in selected for key in row['values']}):
                values = [row['values'][key] for row in selected if key in row['values']]
                try:
                    numeric = [float(value) for value in values]
                except ValueError:
                    metrics[key] = dict(observed_values=sorted(set(values)))
                else:
                    metrics[key] = dict(min=min(numeric), median=statistics.median(numeric),
                                        max=max(numeric), nonzero_samples=sum(x != 0 for x in numeric))
            record['arms'].append(dict(arm=initial.parent.name, sampled_rows=len(selected),
                                       interval_unix=[start, end], metrics=metrics))
        campaigns.append(record)
    args.out.write_text(json.dumps(dict(status='COMPLETE_OBSERVED_COMPARABILITY_ONLY',
        campaigns=campaigns, limits='15-second snapshots, not continuous monitoring. Arm intervals use host inventory-file write timestamps outside timed waves. Earlier unobserved intervals remain unobserved. Power-limit throttling is expected under the fixed cap; no new energy or power-efficiency comparison was performed.'), indent=2) + '\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Quantify the 64K review trigger without replacing full-campaign results."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[3]


def receipt(path):
    path = path.resolve()
    return dict(path=str(path.relative_to(REPO)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def measured(path):
    raw = json.loads(path.read_text())
    waves = [x for x in raw['cases'] if x['scenario']['name'] == 'phase-64k-c1']
    assert [x['repeat'] for x in waves] == [1, 2, 3]
    assert all(x['fully_overlapped_counter_window_valid'] and not x['preemptions'] for x in waves)
    return dict(
        prefill_tps_median=statistics.median(x['native_prefill_compute_tokens_per_s'] for x in waves),
        decode_tps_median=statistics.median(x['native_weighted_decode_tokens_per_s'] for x in waves),
        ttft_s_median=statistics.median(q['ttft_s'] for x in waves for q in x['requests']),
        batch_wall_s_median=statistics.median(x['batch_wall_s'] for x in waves),
        mtp_acceptance=sum(x['speculative_accepted_tokens'] for x in waves) / sum(x['speculative_draft_tokens'] for x in waves),
        fully_overlapped_decode_tps=sum(x['fully_overlapped_generation_tokens'] for x in waves) / sum(x['fully_overlapped_decode_sampled_s'] for x in waves),
        round_equivalent_ms_median=statistics.median(x['fully_overlapped_round_equivalent_ms'] for x in waves),
        waves=[dict(repeat=x['repeat'], decode_tps=x['native_weighted_decode_tokens_per_s'],
                    mtp_acceptance=x['speculative_acceptance'], round_equivalent_ms=x['fully_overlapped_round_equivalent_ms'],
                    output_sha256=x['requests'][0]['output_sha256']) for x in waves])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeat-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists()
    campaign = json.loads((a.repeat_dir / 'campaign.json').read_text())
    assert campaign['status'] == 'COMPLETE_NEW_IMAGE_PERFORMANCE_REPEATS'
    assert campaign['selected'] == ['phase-64k-c1']
    old_summary = REPO / 'benchmarks/results/exl3-review-release-v2/serving-summary.json'
    new_summary = REPO / 'benchmarks/results/vllm-031-upgrade/serving-summary.json'
    old = json.loads(old_summary.read_text()); new = json.loads(new_summary.read_text())
    assert campaign['image_id'] == new['image_id']
    paths = dict(stored_v2=REPO / old['source_review']['raw_results_path'],
                 v031_full=REPO / new['source_review']['raw_results_path'],
                 v031_repeat=REPO / campaign['raw_results_path'])
    assert receipt(paths['stored_v2'])['sha256'] == old['source_review']['raw_results_sha256']
    assert receipt(paths['v031_full'])['sha256'] == new['source_review']['raw_results_sha256']
    assert receipt(paths['v031_repeat'])['sha256'] == campaign['raw_results_sha256']
    data = {name: measured(path) for name, path in paths.items()}
    changes = {name: {key: 100 * (value / data['stored_v2'][key] - 1)
                     for key, value in values.items() if key != 'waves'}
               for name, values in data.items() if name != 'stored_v2'}
    same_outputs = {name: [x['output_sha256'] == y['output_sha256']
                         for x, y in zip(data['v031_full']['waves'], values['waves'])]
                    for name, values in data.items() if name != 'v031_full'}
    report = dict(status='COMPLETE_64K_MEASUREMENTS_REQUIRE_ENGINEERING_REVIEW',
                  image_id=new['image_id'], baseline_regenerated=False, measurements=data,
                  changes_vs_stored_v2_pct=changes, output_hash_matches_v031_full=same_outputs,
                  receipts={**{name: receipt(path) for name, path in paths.items()},
                            'repeat_campaign': receipt(a.repeat_dir / 'campaign.json'),
                            'new_summary': receipt(new_summary), 'old_summary': receipt(old_summary)},
                  scope='Three frozen tasks/repeats per set. The old baseline is historical; the new-only repeat uses a fresh worker. No contemporaneous paired A/B or statistical guarantee. Round-equivalent time includes host/device costs and is not a GPU-only physical-iteration measurement. Original full-campaign results remain primary.')
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], changes=changes, output_matches=same_outputs), indent=2))


if __name__ == '__main__':
    main()

"""Counter decomposition of smaller regressions; not GPU critical-path time."""
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
    cells = {}
    receipts = {}
    for root in args.root:
        path = root / 'campaign.json'
        campaign = json.loads(path.read_text())
        assert campaign['status'] == 'COMPLETE_PAIRED_SCREEN_REVIEW_REQUIRED'
        receipts[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for block in campaign['blocks']:
            maps = {role: {(row['window'], row['mode']): row['result'] for row in rows}
                    for role, rows in block['arms'].items()}
            for key, baseline in maps['baseline'].items():
                candidate = maps['candidate'][key]
                record = {}
                for role, value in [('baseline', baseline), ('candidate', candidate)]:
                    rounds = value['speculative_rounds_sum']
                    record[role] = dict(usable_tokens_per_request_round=value['post_first_completion_tokens'] / rounds,
                                        request_round_equivalent_ms=1000 * value['native']['decode_seconds'] / rounds,
                                        usable_tokens=value['post_first_completion_tokens'],
                                        request_rounds=rounds,
                                        native_request_weighted_decode_tps=value['native_request_weighted_decode_tps'])
                for field in ['usable_tokens_per_request_round', 'request_round_equivalent_ms',
                              'native_request_weighted_decode_tps']:
                    record[field + '_change_pct'] = 100 * (record['candidate'][field] / record['baseline'][field] - 1)
                ratio = (record['candidate']['usable_tokens_per_request_round'] / record['baseline']['usable_tokens_per_request_round']) / (record['candidate']['request_round_equivalent_ms'] / record['baseline']['request_round_equivalent_ms'])
                assert abs(ratio - candidate['native_request_weighted_decode_tps'] / baseline['native_request_weighted_decode_tps']) < 1e-10
                cells.setdefault('/'.join(key), []).append(record)
    summary = {key: dict(paired_blocks=rows, mean_change_pct={field: statistics.mean(row[field + '_change_pct'] for row in rows)
                                                        for field in ['usable_tokens_per_request_round', 'request_round_equivalent_ms', 'native_request_weighted_decode_tps']})
               for key, rows in cells.items()}
    args.out.write_text(json.dumps(dict(status='COMPLETE_COUNTER_DECOMPOSITION_NOT_CAUSAL_ATTRIBUTION',
        cells=summary, receipts_sha256=receipts,
        definitions='Usable post-first completion tokens divided by summed request speculative rounds; summed request decode seconds divided by those same rounds. Their quotient is exactly native request-weighted decode TPS.',
        limits='An algebraic counter decomposition, not a new benchmark or physical GPU group-round measurement. C4 request intervals overlap and can include peer prefill or scheduling stalls; it cannot uniquely attribute changes to kernels, arrivals, acceptance or numeric differences. Full captured round timing is available separately in P0.'), indent=2) + '\n')


if __name__ == '__main__':
    main()

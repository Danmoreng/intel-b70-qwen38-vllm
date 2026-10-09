"""Analyze independent AB/BA blocks, never resample tokens as trials."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def interval(values):
    assert len(values) == 2, 'This analysis is specifically the bounded ABBA screen'
    mean = statistics.mean(values)
    margin = 12.706204736432095 * statistics.stdev(values) / math.sqrt(2)
    return dict(mean=mean, lower=mean-margin, upper=mean+margin,
                independent_blocks=2, method='Student t, df=1; descriptive screening interval')


def analyze(root):
    path = root / 'campaign.json'
    campaign = json.loads(path.read_text())
    assert campaign['status'] == 'COMPLETE_BOUNDED_ABBA_SCREEN'
    assert campaign['order'] == [3, 4, 4, 3]
    assert campaign['restoration']['image_id'] == campaign['production_image']
    assert campaign['restoration']['models_endpoint_healthy']
    weights = campaign['weights']
    assert sum(weights.values()) == 1
    arms = campaign['arms']
    assert all(a['status'] == 'COMPLETE' for a in arms)
    lookup = [{(w['window'], w['cache_mode']): w for w in a['waves']} for a in arms]
    cells, gains = [], {'cold': [], 'warm': []}
    for block, (a, b) in enumerate([(0, 1), (3, 2)]):
        for mode in gains:
            weighted = 0
            for name, weight in weights.items():
                x, y = lookup[a][name, mode], lookup[b][name, mode]
                for row in [x, y]:
                    assert row['native']['preemptions'] == 0
                    assert row['native']['generation_tokens'] == row['concurrency'] * 512
                    assert (row['native']['cached_tokens'] > 0) == (mode == 'warm')
                # The immutable initial helper divided all512 outputs by the
                # post-first decode interval. Standardize absolute rates to
                #511 outputs; ratios are unchanged because both arms use512.
                normalized=[]
                for row in [x,y]:
                    numerator=row['native']['generation_tokens']-row['native']['completed']
                    normalized.append(numerator/row['native']['decode_seconds'])
                gain = normalized[1]/normalized[0] - 1
                weighted += weight * gain
                raw=[]
                for index, depth, row in [(a,3,x),(b,4,y)]:
                    file=root/f'arm-{index}-mtp{depth}'/f"{name}-c{row['concurrency']}-{mode}"/'result.json'
                    full=json.loads(file.read_text())
                    assert full['cache_mode']==mode and len(full['responses'])==row['concurrency']
                    raw.append(statistics.mean(r['wall_s'] for r in full['responses']))
                cells.append(dict(block=block, window=name, mode=mode, weight=weight,
                    mtp3_tps=normalized[0],mtp4_tps=normalized[1],relative_decode_gain=gain,
                    aggregate_client_gain=y['client_aggregate_decode_tps']/x['client_aggregate_decode_tps']-1,
                    wave_output_gain=y['wave_aggregate_output_tps']/x['wave_aggregate_output_tps']-1,
                    mean_client_ttft_change=statistics.mean(y['client_ttft_s'])/statistics.mean(x['client_ttft_s'])-1,
                    mean_client_e2e_change=raw[1]/raw[0]-1,
                    mean_acceptance_length_mtp3=x['mean_acceptance_length'],
                    mean_acceptance_length_mtp4=y['mean_acceptance_length']))
            gains[mode].append(weighted)
    combined = [(x+y)/2 for x, y in zip(gains['cold'], gains['warm'])]
    summaries = {k: interval(v) for k, v in gains.items()}
    summaries['combined_equal_cold_warm_screen'] = interval(combined)
    primary = summaries['combined_equal_cold_warm_screen']
    result = dict(status='BOUNDED_SCREEN_NOT_PRODUCTION_QUALIFICATION',
        campaign_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        image=campaign['image'], weights=weights, block_weighted_gains=gains,
        intervals=summaries, cells=cells,
        counter_definition='Post-first native decode: (generation_tokens - completed_requests) / request decode seconds. Initial raw helper used512 instead of511; correction changes absolute rates by-0.1953125%, leaves every paired rate ratio unchanged, and does not rewrite raw campaigns.',
        clears_point_gain_and_descriptive_interval=primary['mean'] >= .05 and primary['lower'] > 0,
        limitation='Two independent AB/BA pairs only. Cold/warm are correlated within a pair. No token-level or request-level pseudo-replication. Promotion also needs state/capacity/correctness and latency guards.',
        production_restoration=campaign['restoration'])
    (root/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    print(json.dumps(result['intervals'], indent=2))

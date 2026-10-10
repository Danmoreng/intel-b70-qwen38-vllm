#!/usr/bin/env python3
"""Review exact-fixture controls; never automatically approve a release."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from analyze_screen import summarize

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIXTURES = REPO.parent / 'intel-b70-qwen38-vllm/benchmark-results/meaningful-full-profile/run-20260923-201101-w0.00'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    at = (len(ordered) - 1) * fraction
    lo = int(at)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def aggregate(rows):
    requests = [request for row in rows for request in row['requests']]
    elapsed = sum(row['fully_overlapped_decode_sampled_s'] or 0 for row in rows)
    drafted = sum(row['speculative_draft_tokens'] for row in rows)
    rounds = drafted / 3
    post_first = sum(row['post_first_completion_tokens'] for row in rows)
    return {
        'native_decode_tps': statistics.median(row['native_weighted_decode_tokens_per_s'] for row in rows),
        'client_aggregate_decode_tps': statistics.median(row['aggregate_decode_wave_tokens_per_s'] for row in rows),
        'fully_overlapped_decode_tps_unfiltered_counter_rate': sum(row['fully_overlapped_generation_tokens'] or 0 for row in rows) / elapsed,
        'prefill_tps': statistics.median(row['native_prefill_compute_tokens_per_s'] for row in rows),
        'ttft_median_s': statistics.median(request['ttft_s'] for request in requests),
        'ttft_p95_s': percentile([request['ttft_s'] for request in requests], .95),
        'batch_wall_median_s': statistics.median(row['batch_wall_s'] for row in rows),
        'slowest_request_per_wave_median_s': statistics.median(max(request['wall_s'] for request in row['requests']) for row in rows),
        'request_wall_p95_s': percentile([request['wall_s'] for request in requests], .95),
        'mtp_acceptance': sum(row['speculative_accepted_tokens'] for row in rows) / drafted,
        'usable_tokens_per_request_round_equivalent': post_first / rounds,
        'request_round_equivalent_ms': 1000 * sum(row['native_decode_seconds'] for row in rows) / rounds,
    }


def load_arm(root, arm, selected):
    path = root / arm['results_path']
    assert arm['status'] == 'COMPLETE' and sha(path) == arm['results_sha256']
    data = json.loads(path.read_text())
    assert {row['scenario']['name'] for row in data['cases']} == selected
    grouped = {name: [] for name in selected}
    verified = []
    for row in data['cases']:
        scenario, repeat = row['scenario'], row['repeat']
        assert row['preemptions'] == 0 and row['prefill_recompute_excess'] == 0
        assert row['prompt_tokens_cached'] == 0
        assert row['all_prompt_counts_match'] and row['all_completion_counts_exact'] and row['all_finish_reasons_length']
        assert len(row['requests']) == scenario['concurrency'] and scenario['output_tokens'] == 1024
        assert row['fully_overlapped_decode_sampled_s'] > 0
        folder = f"{scenario['name']}-r{repeat}"
        for index, request in enumerate(row['requests'], 1):
            assert request['usage']['completion_tokens'] == 1024 and request['finish_reason'] == 'length'
            prompt = path.parent / folder / f'prompt-{index}.txt'
            original_prompt = FIXTURES / folder / prompt.name
            request_path = path.parent / folder / f'request-{index}.json'
            original_request = FIXTURES / folder / request_path.name
            assert prompt.read_bytes() == original_prompt.read_bytes()
            # Compare values, allowing harmless JSON key-order differences.
            assert json.loads(request_path.read_text()) == json.loads(original_request.read_text())
            verified.append({'case': scenario['name'], 'repeat': repeat, 'request': index,
                             'prompt_sha256': sha(prompt), 'request_fixture_sha256': sha(original_request)})
        grouped[scenario['name']].append(row)
    for rows in grouped.values():
        assert len(rows) == rows[0]['scenario']['repeats']
        assert sorted(row['repeat'] for row in rows) == list(range(1, len(rows) + 1))
    return grouped, verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    campaign_path = root / 'campaign.json'
    campaign = json.loads(campaign_path.read_text())
    assert campaign['status'] == 'COMPLETE_NAMED_FROZEN_REFERENCES_REVIEW_REQUIRED'
    assert campaign['restoration']['image_id'] == campaign['baseline'] and campaign['restoration']['models_endpoint_healthy']
    assert sha(HERE / 'confirm_frozen_triggers.py') == campaign['source_sha256']
    assert sha(REPO / 'scripts/current-profile-benchmark.py') == campaign['benchmark_sha256']
    comparison = REPO / 'benchmarks/results/b70-stack-20261010/benchmark-comparison.json'
    assert sha(comparison) == campaign['comparison_sha256']
    selected = set(campaign['cases'])
    assert len(campaign['blocks']) == campaign['independent_blocks'] == 3
    cells = {name: {'paired_blocks': [], 'metric_effects': {}, 'paired_fixture_geometric_effects': {},
                    'per_fixture_effects': {}} for name in sorted(selected)}
    sources = []
    for block in campaign['blocks']:
        assert block['order'] == (['baseline', 'candidate'] if block['index'] % 2 == 0 else ['candidate', 'baseline'])
        arms = {}
        fixtures = {}
        for role in ['baseline', 'candidate']:
            arm = block['arms'][role]
            assert arm['num_blocks'] == 200 and arm['kv_pool_bytes'] == 11141120000
            inventory = root / f"block-{block['index']}-{role}/inventory-initial.json"
            assert sha(inventory) == arm['inventory_sha256']
            arms[role], fixtures[role] = load_arm(root, arm, selected)
            sources.append({'block': block['index'], 'role': role, 'results_path': arm['results_path'],
                            'results_sha256': arm['results_sha256'], 'inventory_sha256': arm['inventory_sha256']})
        assert fixtures['baseline'] == fixtures['candidate']
        for name in sorted(selected):
            cell = cells[name]
            values = {role: aggregate(arms[role][name]) for role in arms}
            baseline_rows = {row['repeat']: row for row in arms['baseline'][name]}
            candidate_rows = {row['repeat']: row for row in arms['candidate'][name]}
            hashes = []
            fixture_ratios = {}
            fixture_values = []
            for repeat in sorted(baseline_rows):
                per_fixture = {role: aggregate([rows[repeat]]) for role, rows in
                               [('baseline', baseline_rows), ('candidate', candidate_rows)]}
                percentages = {metric: 100 * (per_fixture['candidate'][metric] / value - 1)
                               for metric, value in per_fixture['baseline'].items()}
                fixture_values.append({'repeat': repeat, 'values': per_fixture, 'change_pct': percentages})
                effects = cell['per_fixture_effects'].setdefault(str(repeat), {})
                for metric, effect in percentages.items():
                    effects.setdefault(metric, []).append(effect)
                    fixture_ratios.setdefault(metric, []).append(math.log(per_fixture['candidate'][metric] / per_fixture['baseline'][metric]))
                for index, (old, new) in enumerate(zip(baseline_rows[repeat]['requests'], candidate_rows[repeat]['requests']), 1):
                    hashes.append({'repeat': repeat, 'request': index, 'baseline_output_sha256': old['output_sha256'],
                                   'candidate_output_sha256': new['output_sha256'], 'equal': old['output_sha256'] == new['output_sha256']})
            overlap_windows = {role: [{'repeat': row['repeat'],
                'round_accounting_valid': row['fully_overlapped_counter_window_valid'],
                'issues': row['fully_overlapped_counter_window_issues'],
                'token_accounting_residual': row['fully_overlapped_token_accounting_residual'],
                'generation_tokens': row['fully_overlapped_generation_tokens'],
                'sampled_s': row['fully_overlapped_decode_sampled_s'],
                'round_equivalent_ms': row['fully_overlapped_round_equivalent_ms']}
                for row in arms[role][name]] for role in arms}
            cell['paired_blocks'].append({'block': block['index'], 'order': block['order'], 'values': values,
                                         'per_fixture_values': fixture_values,
                                         'overlap_counter_windows': overlap_windows,
                                         'output_comparisons': hashes, 'fixtures': [f for f in fixtures['baseline'] if f['case'] == name]})
            for metric, value in values['baseline'].items():
                cell['metric_effects'].setdefault(metric, []).append(100 * (values['candidate'][metric] / value - 1))
            for metric, logs in fixture_ratios.items():
                cell['paired_fixture_geometric_effects'].setdefault(metric, []).append(100 * math.expm1(statistics.mean(logs)))
    for cell in cells.values():
        cell['metric_effects'] = {metric: summarize(effects) for metric, effects in cell['metric_effects'].items()}
        cell['paired_fixture_geometric_effects'] = {metric: summarize(effects) for metric, effects in cell['paired_fixture_geometric_effects'].items()}
        cell['per_fixture_effects'] = {repeat: {metric: summarize(effects) for metric, effects in metrics.items()}
                                     for repeat, metrics in cell['per_fixture_effects'].items()}
    result = {'status': 'EXACT_FIXTURE_PAIRED_CONTROLS_REVIEW_REQUIRED_NOT_RELEASE_APPROVAL',
              'images': {role: campaign[role] for role in ['baseline', 'candidate']},
              'analysis_sha256': sha(Path(__file__)),
              'bootstrap_methods_sha256': sha(HERE / 'analyze_screen.py'),
              'campaign_sha256': sha(campaign_path), 'independent_blocks': 3, 'cells': cells, 'sources': sources,
              'definitions': 'Each original fixture retains all its original repeats, warmup, prompt bytes, request values and 1024-token budget. metric_effects compares within-arm medians to match the historical table convention, except the unfiltered fully-overlapped counter rate uses summed generation tokens / summed seconds and acceptance uses summed counters. paired_fixture_geometric_effects pairs each exact repeat fixture before averaging log-ratios with equal fixture weights inside each startup block; per_fixture_effects retains each fixture separately across startup blocks. Both views are reported; a favorable median ratio does not erase a losing paired-fixture/client metric. Every window is retained, including original invalid round-accounting flags, residuals and unavailable round timings; no adverse wave is filtered. Percentile bootstrap resamples the three independent paired startup blocks, not requests or tokens. Positive TPS is better; positive seconds is worse. No weighted speed headline from this selected adverse panel.',
              'limits': 'Three paired blocks have low power. Bootstrap intervals are descriptive, not a formal equivalence or population p95 guarantee. E2E p95 is an empirical request percentile within the finite original repeats. Invalid overlap round-accounting windows must not support a validated physical round-time claim; their unfiltered counter rate is an observation with that qualification. Request round-equivalents divide summed full-request draft tokens by three and include request overlap; they are not physical C4 GPU cycle times. Different valid generated histories may alter acceptance and serving speed, which remains a release consideration. Historical adverse results are retained in benchmark-comparison.json.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: cell['metric_effects'] for name, cell in cells.items()}, indent=2))


if __name__ == '__main__':
    main()

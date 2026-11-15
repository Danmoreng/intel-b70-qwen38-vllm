#!/usr/bin/env python3
"""Validate paired targeted measurements without approving a production release."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

FLAGS = ('all_outputs_nonempty', 'all_prompt_counts_match',
         'all_completion_counts_exact', 'all_finish_reasons_length',
         'admission_expectation_met')
TARGETS = ('phase-4k-c1', 'concurrency-2k-c2')


def load_session(root, session):
    folder = root / session['label']
    campaign = json.loads((folder / 'campaign.json').read_text())
    assert session['status'] == 'PASS' and campaign['status'] == 'PASS_REQUESTED_STAGES'
    runs = list((folder / 'serving').glob('run-*'))
    assert len(runs) == 1
    rows = {}
    for path in runs[0].glob('*/summary.json'):
        value = json.loads(path.read_text())
        assert all(value[flag] for flag in FLAGS), path
        assert value['preemptions'] == 0, path
        assert value['prompt_tokens_cached'] == 0, ('Cold comparison unexpectedly reused KV', path)
        key = (value['scenario']['name'], value['repeat'])
        assert key not in rows
        rows[key] = (value, path.parent)
    assert len(rows) == (8 if session['regime'] == 'fresh' else 11)
    return rows


def summarize(rows):
    def med(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return median(values) if values else None
    output = sum(r['completion_tokens'] for r in rows)
    post = sum(r['post_first_completion_tokens'] for r in rows)
    wall = sum(r['batch_wall_s'] for r in rows)
    native_decode = sum(r['native_decode_seconds'] for r in rows)
    native_prefill = sum(r['native_prefill_seconds'] for r in rows)
    drafts = sum(r['speculative_draft_tokens'] for r in rows)
    return {
        'waves': len(rows), 'requests': sum(len(r['requests']) for r in rows),
        'wall_median_s': med('batch_wall_s'),
        'wall_min_s': min(r['batch_wall_s'] for r in rows),
        'wall_max_s': max(r['batch_wall_s'] for r in rows),
        'wall_total_s': wall, 'output_tokens': output,
        'output_tokens_per_wall_second': output / wall,
        'ttft_median_s': median(r['request_ttft_s']['median'] for r in rows),
        'native_decode_weighted_tokens_per_s': post / native_decode,
        'native_prefill_weighted_tokens_per_s': sum(r['prefill_tokens_computed'] for r in rows) / native_prefill,
        'speculative_acceptance': sum(r['speculative_accepted_tokens'] for r in rows) / drafts,
        'round_equivalent_median_ms': med('fully_overlapped_round_equivalent_ms'),
        'preemptions': sum(r['preemptions'] for r in rows),
        'cached_prompt_tokens': sum(r['prompt_tokens_cached'] for r in rows),
    }


def compare(old, new):
    left, right = summarize(old), summarize(new)
    changes = {key: 100 * (right[key] / left[key] - 1)
               for key in ('wall_median_s', 'wall_total_s', 'output_tokens_per_wall_second',
                           'ttft_median_s', 'native_decode_weighted_tokens_per_s',
                           'native_prefill_weighted_tokens_per_s', 'round_equivalent_median_ms')
               if left[key] is not None and right[key] is not None and left[key] != 0}
    return {'baseline': left, 'candidate': right, 'changes_pct': changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    campaign = json.loads((root / 'campaign.json').read_text())
    assert campaign['status'] == 'PASS_ALL_FOUR_SESSIONS'
    sessions = campaign['sessions']
    assert len(sessions) == 4
    contract = json.loads((root / 'frozen-contract.json').read_text())
    repo = Path(__file__).resolve().parents[3]
    for name, expected in contract['files_sha256'].items():
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == expected, ('Frozen measurement source changed', name)
    for session in sessions:
        recorded = json.loads((root / session['label'] / 'campaign.json').read_text())
        assert recorded['image_id'] == contract['images'][session['arm']]
    data = {(s['regime'], s['arm']): load_session(root, s) for s in sessions}
    assert len(data) == 4
    results, pairs, pooled = [], [], {'v7': [], 'R12': []}
    matched_requests = 0
    for regime in ('fresh', 'after-128k'):
        old, new = data[(regime, 'v7')], data[(regime, 'R12')]
        assert set(old) == set(new)
        names = sorted({name for name, _ in old})
        for name in names:
            keys = sorted(key for key in old if key[0] == name)
            result = {'regime': regime, 'scenario': name,
                      **compare([old[k][0] for k in keys], [new[k][0] for k in keys])}
            results.append(result)
        for key in sorted(old):
            left, left_dir = old[key]
            right, right_dir = new[key]
            assert left['scenario'] == right['scenario']
            assert len(left['requests']) == len(right['requests'])
            equal_outputs = 0
            for i, (u, v) in enumerate(zip(left['requests'], right['requests']), 1):
                assert (left_dir / f'prompt-{i}.txt').read_bytes() == (right_dir / f'prompt-{i}.txt').read_bytes()
                assert json.loads((left_dir / f'request-{i}.json').read_text()) == json.loads((right_dir / f'request-{i}.json').read_text())
                assert u['usage']['prompt_tokens'] == v['usage']['prompt_tokens']
                assert u['usage']['completion_tokens'] == v['usage']['completion_tokens'] == 1024
                equal_outputs += u['output_sha256'] == v['output_sha256']
                matched_requests += 1
            pairs.append({'regime': regime, 'scenario': key[0], 'repeat': key[1],
                          'baseline_wall_s': left['batch_wall_s'], 'candidate_wall_s': right['batch_wall_s'],
                          'baseline_prefill_s': left['native_prefill_seconds'], 'candidate_prefill_s': right['native_prefill_seconds'],
                          'baseline_acceptance': left['speculative_acceptance'], 'candidate_acceptance': right['speculative_acceptance'],
                          'identical_output_hashes': equal_outputs, 'requests': len(left['requests'])})
            if key[0] in TARGETS:
                pooled['v7'].append(left)
                pooled['R12'].append(right)
    assert matched_requests == 25
    combined = []
    for name in TARGETS:
        combined.append({'scenario': name, **compare([r for r in pooled['v7'] if r['scenario']['name'] == name],
                                                     [r for r in pooled['R12'] if r['scenario']['name'] == name])})
    report = {
        'status': 'ALL38_WAVES_VALID_TARGETED_FOLLOWUP_MEASURED_NOT_RELEASE_APPROVAL',
        'actual_requests': 2 * matched_requests,
        'paired_request_payloads_and_prompt_bytes_verified': matched_requests,
        'all38_measured_waves_valid': True, 'preemptions': 0,
        'condition_comparisons': results, 'combined_target_comparisons': combined,
        'paired_waves': pairs,
        'limits': 'Targeted follow-up is not a replacement full70-wave score. All attempts retained. Each condition has one session per image; fixed finite fixtures are not independent population samples. The128K prologue reproduces the immediately preceding long-context sequence, not the full historical36-wave prefix. Round-equivalent timing includes host/device costs and is not pure GPU kernel latency. No production promotion is performed by this tool.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'combined_target_comparisons': combined}, indent=2))


if __name__ == '__main__':
    main()

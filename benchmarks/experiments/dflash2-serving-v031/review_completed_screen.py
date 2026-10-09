#!/usr/bin/env python3
"""Review a completed screen whose controller failed on a postcheck field name.

Keep the failed controller receipt unchanged. This reads the existing frozen
data and does not contact the GPU, repeat requests or suppress counter issues.
"""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--running-source', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    original_path = root / 'campaign.json'
    original = json.loads(original_path.read_text())
    assert original['status'] == 'FAILED'
    assert original['error'] == "KeyError('completion_tokens')"
    assert sha(args.running_source) == original['sources']['run_serving.py']
    assert "q['completion_tokens']==1024" in args.running_source.read_text()
    assert original['restoration']['models_endpoint_healthy']
    assert original['restoration']['image_id'] == original['production_image']
    paths = list(root.glob('speed/*/results.json'))
    assert len(paths) == 1
    raw_path = paths[0]
    raw = json.loads(raw_path.read_text())
    expected = {(name, repeat) for name in ['phase-4k-c1', 'phase-32k-c1', 'phase-64k-c1'] for repeat in [1, 2, 3]}
    assert len(raw['cases']) == 9
    assert {(c['scenario']['name'], c['repeat']) for c in raw['cases']} == expected
    assert raw['manifest']['container']['image'] == original['image']
    assert raw['manifest']['container']['num_speculative_tokens'] == original['depth']
    assert raw['manifest']['container']['max_model_len'] == 96000
    for case in raw['cases']:
        assert case['preemptions'] == 0
        assert case['all_prompt_counts_match'] and case['all_completion_counts_exact']
        assert case['all_finish_reasons_length'] and case['all_outputs_nonempty']
        assert len(case['requests']) == 1
        assert case['requests'][0]['usage']['completion_tokens'] == 1024
    assert json.loads((root / 'prefix-state.json').read_text())['status'] == 'PASS'
    identity = json.loads((root / 'worker-attestation.json').read_text())['results']
    assert len(identity) == 1 and identity[0]['status'] == 'PASS_ACTUAL_FP16_DFLASH2_WORKER'
    assert identity[0]['num_speculative_tokens'] == original['depth']
    assert identity[0]['target_dtype'] == identity[0]['draft_dtype'] == 'torch.float16'
    audits = list((root / 'context-96000/loader').glob('loader-*.json'))
    assert len(audits) == 1
    audit = json.loads(audits[0].read_text())
    assert audit['status'] == 'PASS'
    assert audit['expected_modules'] == audit['loaded_modules'] == 401
    assert audit['mtp_expected'] == audit['mtp_loaded'] == 0
    smoke = json.loads((root / 'smoke.json').read_text())
    assert smoke['accepted'] > 0 and smoke['drafted'] > 0
    reviewed = dict(original)
    reviewed.update(status='PASS_UNQUANTIZED_DFLASH_STARTUP_ACCEPTANCE_AND_SPEED', active_context_limit=96000,
                    benchmark_results=str(raw_path.relative_to(REPO)), benchmark_sha256=sha(raw_path),
                    postcheck_review=dict(original_campaign_sha256=sha(original_path), original_controller_status=original['status'],
                                         original_error=original['error'], reviewer_sha256=sha(Path(__file__)),
                                         note='Nine waves completed. Original postcheck read completion_tokens at the wrong nesting level; verified existing usage.completion_tokens. No GPU rerun.',
                                         sampled_window_issues=[dict(scenario=c['scenario']['name'], repeat=c['repeat'], issues=c['fully_overlapped_counter_window_issues'], residual=c['fully_overlapped_token_accounting_residual']) for c in raw['cases'] if not c['fully_overlapped_counter_window_valid']],
                                         sampled_round_equivalent_metrics_qualified=False))
    target = root / 'campaign-reviewed.json'
    assert not target.exists()
    target.write_text(json.dumps(reviewed, indent=2) + '\n')
    print(target)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Export whitelisted evidence for the exact final maintenance candidate.

No production write, no automatic performance review, no GPU operations.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RAW = REPO / 'benchmark-results/b70-stack-20261010'
PUBLIC = REPO / 'benchmarks/results/b70-stack-20261010'
CFG = REPO / 'config/experiments/b70-stack-20261010'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', default='E01-final-controller.json')
    parser.add_argument('--operations', default='E01-operations-final')
    args = parser.parse_args()
    operation_root = RAW / args.operations
    receipts = {}

    def read(path):
        receipts[str(path.relative_to(REPO))] = sha(path)
        return json.loads(path.read_text())

    release = read(CFG / 'production_image.json')
    image, runtime, policy = [release[k] for k in ('image_id', 'runtime_image_id', 'policy_sha256')]
    assert sha(CFG / 'production_policy.json') == policy
    metadata = read(RAW / 'E01-final-metadata/image.json')
    assert metadata['image_id'] == image and metadata['runtime_image_id'] == runtime
    assert metadata['rootfs'] == release['rootfs_metadata_attestation']['rootfs']
    controller = read(RAW / args.controller)
    assert controller['status'] == 'COMPLETE_FINAL_OPERATIONS_AND_FROZEN_SERVING_REVIEW_REQUIRED'
    for name, digest in controller['sources_sha256'].items():
        assert sha(REPO / name) == digest
        receipts[name] = digest
    gates = read(RAW / 'E01-gpu-gates/campaign.json')
    quality = read(RAW / 'E01-quality/numerical-review.json')
    replay = read(RAW / 'E01-quality/replay-campaign.json')
    screen = read(PUBLIC / 'E01-maintenance-serving-review.json')
    assert all(x['image_id'] == runtime for x in [gates, quality, replay])
    assert all(x['status'] == 'PASS' for x in gates['commands'])
    assert quality['status'] == 'PASS_REVIEWED_CHANGED_STACK_NUMERICS'
    assert replay['status'] == 'PASS_NEW_IMAGE_MATCHED_STATE_ORIGINAL_TOLERANCES'
    assert screen['status'] == 'PASS_REVIEWED_MAINTENANCE_SERVING_GATES'
    assert screen['runtime_image_id'] == runtime
    assert read(RAW / 'E01-quality/campaign.json')['status'] == 'FAILED'
    original = read(RAW / 'E01-serving-screen/campaign.json')
    boundary = original['capacity']['near-limit']
    assert all(boundary['usage'][key] == value for key, value in
               dict(prompt_tokens=261120, completion_tokens=1024, total_tokens=262144).items())
    assert boundary['native']['preemptions'] == 0
    operations = read(operation_root / 'operations.json')
    assert operations['status'] == 'PASS_NEW_IMAGE_OPERATIONAL_GATES'
    assert operations['image_id'] == image and operations['policy_sha256'] == policy
    assert operations['restarted_inventory']['num_blocks'] == 200
    assert operations['restarted_inventory']['kv_pool_bytes'] == 11141120000
    assert operations['cases']['c4-long']['native']['preemptions'] == 0
    for name, decoders in [('mixed-2', 1), ('mixed-4', 3)]:
        case = operations['cases'][name]
        assert case['status'] == 'COMPLETE' and case['decoders'] == decoders
        assert case['total_admitted'] == decoders + 1
        assert case['native']['preemptions'] == 0
        assert case['during_incoming_prefill_burst_gaps_s']['count'] > 0
    assert operations['confirmed_abort_trace_occurrences'] > 0
    assert operations['cases']['worker-restart']['status'] == 'PASS'
    pressure = operations['c16_pressure_comparison']
    if operations.get('c16_reference_review_sha256'):
        reviewed_path = PUBLIC / 'E01-c16-matched-review.json'
        assert sha(reviewed_path) == operations['c16_reference_review_sha256'] == controller['c16_review_sha256']
        reviewed = read(reviewed_path)
        assert reviewed['status'] == 'PASS_REVIEWED_NO_REPRODUCED_C16_REGRESSION'
        assert pressure['baseline_preemptions'] == reviewed['current_baseline_preemption_median']
        assert pressure['historical_baseline_preemptions'] == 2
        # Retain failed history as checksummed input and a sanitized public
        # attempts receipt; never copy its synthetic raw API/media responses.
        assert read(RAW / 'E01-final-controller-r2.json')['status'] == 'FAILED'
        assert read(RAW / 'E01-operations-final-r2/operations.json')['status'] == 'FAILED'
        read(PUBLIC / 'E01-operations-attempts.json')
    assert pressure['candidate_preemptions'] <= pressure['baseline_preemptions']
    assert pressure['candidate_prefill_recompute_excess'] <= pressure['baseline_prefill_recompute_excess']
    coverage = read(operation_root / 'runtime-sources/SOURCE_GUARD_COVERAGE.json')
    assert coverage['covered_count'] == coverage['required_count'] == 16
    for name, digest in coverage['files_sha256'].items():
        assert sha(operation_root / 'runtime-sources' / name) == digest
    assert read(operation_root / 'runtime-sources/CAPTURE_IDENTITY.json')['image_id'] == image
    serving = read(RAW / 'E01-serving-final/campaign.json')
    assert serving['status'] == 'COMPLETE_REVIEW_SERVING_MATRIX'
    assert serving['image_receipt']['image_id'] == image
    spec = importlib.util.spec_from_file_location('stack_summary', REPO / 'scripts/summarize-readme-benchmarks.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.summarize(RAW / 'E01-serving-final', serving_only=True)
    assert summary['image_id'] == image
    assert summary['source_review']['waves'] == 70 and summary['source_review']['requests'] == 124
    write(PUBLIC / 'serving-summary.json', summary)
    old = read(REPO / 'benchmarks/results/vllm-031-upgrade/serving-summary.json')
    previous = {x['name']: x for x in old['source_review']['scenario_results']}
    rows = []
    for new in summary['source_review']['scenario_results']:
        before = previous[new['name']]
        row = dict(name=new['name'], concurrency=new['concurrency'])
        for key in ['prefill_tps_median', 'decode_tps_median', 'ttft_s_median',
                    'batch_wall_s_median', 'fully_overlapped_decode_tps', 'mtp_acceptance']:
            x, y = before.get(key), new.get(key)
            row[key] = dict(stored_v031=x, onednn3134=y,
                            change_pct=100 * (y / x - 1) if x and y is not None else None)
        row['review_trigger'] = bool((row['decode_tps_median']['change_pct'] or 0) < -3 or
                                    (row['ttft_s_median']['change_pct'] or 0) > 5 or
                                    (row['batch_wall_s_median']['change_pct'] or 0) > 5 or
                                    (row['prefill_tps_median']['change_pct'] or 0) < -5 or
                                    (row['fully_overlapped_decode_tps']['change_pct'] or 0) < -3)
        rows.append(row)
    comparison = dict(status='COMPLETE_STORED_V031_COMPARISON_REVIEW_REQUIRED',
                      baseline_regenerated=False, baseline_image_id=old['image_id'],
                      candidate_image_id=image, scenarios=rows,
                      scope='Same frozen workload. Historical full-matrix medians are review triggers, not a paired confidence interval. Independent paired candidate screens supply contemporaneous evidence. No performance win is asserted.')
    write(PUBLIC / 'benchmark-comparison.json', comparison)
    # These JSONs contain counters, hashes and fixed synthetic operational checks.
    # Do not export campaign arm responses, prompt IDs, SSE, logits or container env.
    public_ops = {key: value for key, value in operations.items()
                  if key not in ('cases', 'restoration')}
    public_ops['cases'] = {name: {key: value for key, value in case.items()
                                if key in ('status', 'usage', 'native', 'request_sha256',
                                           'preemption_policy', 'purpose', 'unique_images',
                                           'pixels_each', 'fixture_sha256', 'loader',
                                           'decoders', 'total_admitted', 'wall_s',
                                           'incoming_prefill_client_s',
                                           'during_incoming_prefill_burst_gaps_s')}
                           for name, case in operations['cases'].items()}
    for name in ['mixed-2', 'mixed-4']:
        public_ops['cases'][name]['purpose'] = 'Actual incoming cold49K prefill overlapping existing 4096-output-token decoders on this exact image. SSE burst gaps are client observations, not per-token GPU latency or a paired ABBA performance comparison.'
    public_ops['source_raw_sha256'] = sha(operation_root / 'operations.json')
    write(PUBLIC / 'operations.json', public_ops)
    for section in ['source_review']:
        read(REPO / summary[section]['raw_results_path'])
    for item in summary['source_review'].get('supplementary_prefix_runs', []):
        if 'raw_results_path' in item:
            read(REPO / item['raw_results_path'])
    exports = {}
    # The complete frozen serving counters are safe: requests have hashes and
    # timing, not output text or input token arrays.
    for path in sorted((RAW / 'E01-serving-final').glob('*/*/results.json')):
        read(path)
        destination = PUBLIC / 'evidence' / path.relative_to(RAW)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        exports[str(path.relative_to(REPO))] = dict(path=str(destination.relative_to(REPO)), sha256=sha(destination))
    for path in [RAW / 'E01-final-metadata/image.json', RAW / args.controller,
                 RAW / 'E01-final-controller.json', RAW / 'E01-operations-final/operations.json',
                 RAW / 'E01-quality/numerical-review.json', RAW / 'E01-quality/replay-campaign.json',
                 RAW / 'E01-gpu-gates/campaign.json',
                 operation_root / 'runtime-sources/SOURCE_GUARD_COVERAGE.json',
                 operation_root / 'runtime-sources/CAPTURE_IDENTITY.json']:
        read(path)
        destination = PUBLIC / 'evidence' / path.relative_to(RAW)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        exports[str(path.relative_to(REPO))] = dict(path=str(destination.relative_to(REPO)), sha256=sha(destination))
    for name, digest in coverage['files_sha256'].items():
        path = operation_root / 'runtime-sources' / name
        destination = PUBLIC / 'evidence' / path.relative_to(RAW)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        assert sha(destination) == digest
        exports[str(path.relative_to(REPO))] = dict(path=str(destination.relative_to(REPO)), sha256=digest)
    assessment = dict(status='PASS_EXACT_IMAGE_RELEASE_GATES', image_id=image,
                      policy_sha256=policy, runtime_parent_image_id=runtime,
                      source_commit=release['source_commit'],
                      runtime_artifacts_sha256=release['runtime_artifacts_sha256'],
                      rootfs_metadata_attestation=metadata,
                      numerical_review=quality, independent_paired_screen=screen,
                      capacity=dict(runtime_image_id=runtime, rootfs_identical_final=image, evidence=boundary),
                      operations=dict(path='benchmarks/results/b70-stack-20261010/operations.json',
                                      sha256=sha(PUBLIC / 'operations.json')),
                      source_guard_coverage=dict(required=16, covered=16),
                      serving=dict(scenarios=20, waves=70, requests=124, preemptions=0),
                      receipts_sha256=receipts, public_receipt_mapping=exports,
                      limitations=['Maintenance candidate, not an evidenced decode speedup.',
                                   'The exact short-array equality failure is retained; numerical review uses original tolerances and finite independent/state/quality gates.',
                                   'The 262144 total-token capacity gate ran on the identical runtime parent; final image changes metadata only.',
                                   'One maximum-context request and C16 admission pressure do not establish 16 simultaneous maximum contexts.',
                                   'The retained C16 attempt failed against an archived count of two. Three contemporaneous matched pairs establish the reviewed current pressure reference; zero-preemption cases retain their zero gate.',
                                   'Cold compiler state gives 199 blocks on both images; subsequent normal starts give the original 200 blocks. No KV override.',
                                   'Source64 historical cross-version acceptance cause remains inconclusive; no sampler/state patch is justified.',
                                   'The older power matrix was not rerun. 180W remains selected from its measured efficiency optimum.'])
    write(PUBLIC / 'assessment.json', assessment)
    triggers = [x['name'] for x in rows if x['review_trigger']]
    write(PUBLIC / 'final-review-pending.json', dict(status='REVIEW_REQUIRED_NOT_PRODUCTION_APPROVAL',
          image_id=image, policy_sha256=policy, performance_review_triggers=triggers,
          immediate_rollback_image_id=old['image_id']))
    print(json.dumps(dict(status='FINAL_REVIEW_REQUIRED', performance_review_triggers=triggers)))


if __name__ == '__main__':
    main()

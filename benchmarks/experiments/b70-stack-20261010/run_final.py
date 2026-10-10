#!/usr/bin/env python3
"""Freeze only a reviewed maintenance survivor, then test its exact image.

This controller does not promote production. All GPU children restore the
original qualified image, and their receipts remain distinct from this review.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = REPO / 'benchmark-results/b70-stack-20261010'
PUBLIC = REPO / 'benchmarks/results/b70-stack-20261010'
CFG = REPO / 'config/experiments/b70-stack-20261010'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serving-review-sha256', required=True)
    parser.add_argument('--resume-metadata', action='store_true')
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--c16-review-sha256')
    args = parser.parse_args()
    review_path = PUBLIC / 'E01-maintenance-serving-review.json'
    assert sha(review_path) == args.serving_review_sha256
    review = json.loads(review_path.read_text())
    assert review['status'] == 'PASS_REVIEWED_MAINTENANCE_SERVING_GATES'
    for item in review['receipts'].values():
        assert sha(REPO / item['path']) == item['sha256']
    numerical = json.loads((ROOT / 'E01-quality/numerical-review.json').read_text())
    assert numerical['status'] == 'PASS_REVIEWED_CHANGED_STACK_NUMERICS'
    assert numerical['image_id'] == review['runtime_image_id']
    assert args.attempt >= 1 and (not args.resume_metadata or args.attempt > 1)
    if args.attempt >= 3:
        # One prospectively bounded final-operations retry after all three
        # matched reference pairs. The pressure case compares measured current
        # baseline counts/work; zero-preemption cases keep their zero gate.
        assert args.attempt == 3 and args.resume_metadata
        pressure_path = PUBLIC / 'E01-c16-matched-review.json'
        assert args.c16_review_sha256 and sha(pressure_path) == args.c16_review_sha256
        pressure = json.loads(pressure_path.read_text())
        assert pressure['status'] == 'PASS_REVIEWED_NO_REPRODUCED_C16_REGRESSION'
        assert pressure['runtime_image_id'] == review['runtime_image_id']
        for item in pressure['receipts'].values():
            assert sha(REPO / item['path']) == item['sha256']
        assert pressure['final_operations_retry_limit'] == 1
        assert pressure['historical_preemption_count'] == 2
    suffix = '' if args.attempt == 1 else f'-r{args.attempt}'
    out = ROOT / ('E01-final-controller' + suffix + '.json')
    assert not out.exists()
    state = dict(status='RUNNING', started_unix=time.time(),
                 serving_review_sha256=args.serving_review_sha256,
                 runtime_image_id=review['runtime_image_id'], commands=[],
                 attempt=args.attempt,
                 c16_review_sha256=args.c16_review_sha256,
                 sources_sha256={str(p.relative_to(REPO)): sha(p) for p in
                                 [Path(__file__), HERE / 'freeze_release.py',
                                  HERE / 'run_operations.py',
                                  REPO / 'scripts/run-exl3-review-serving.py',
                                  REPO / 'scripts/current-profile-benchmark.py',
                                  REPO / 'scripts/exl3_candidate_worker.py',
                                  REPO / 'scripts/summarize-readme-benchmarks.py']})

    def save():
        out.write_text(json.dumps(state, indent=2) + '\n')

    def run(name, argv):
        row = dict(name=name, argv=argv, status='RUNNING', started_unix=time.time())
        state['commands'].append(row)
        save()
        print('START', name, flush=True)
        with (ROOT / (name + '-controller.log')).open('w') as log:
            subprocess.run(argv, check=True, cwd=REPO, stdout=log,
                           stderr=subprocess.STDOUT)
        row.update(status='PASS', finished_unix=time.time())
        save()
        print('PASS', name, flush=True)

    try:
        if args.resume_metadata:
            previous = json.loads((ROOT / 'E01-final-controller.json').read_text())
            failed = json.loads((ROOT / 'E01-operations-final/operations.json').read_text())
            assert previous['status'] == 'FAILED' and previous['commands'][0]['status'] == 'PASS'
            assert failed['status'] == 'FAILED' and not failed['cases']
            assert 'not in the subpath' in failed['error']
            metadata = json.loads((ROOT / 'E01-final-metadata/image.json').read_text())
            inspected = json.loads(subprocess.check_output(['docker', 'image', 'inspect',
                                    metadata['image_id'], metadata['runtime_image_id']], text=True))
            assert inspected[0]['RootFS'] == inspected[1]['RootFS'] == metadata['rootfs']
            assert metadata['runtime_image_id'] == review['runtime_image_id']
            assert metadata['policy_sha256'] == sha(CFG / 'production_policy.json')
            state['retry_provenance'] = dict(previous_controller_sha256=sha(ROOT / 'E01-final-controller.json'),
                failed_operations_sha256=sha(ROOT / 'E01-operations-final/operations.json'),
                explanation='Reporting path failure after first cold-start inventory/409+8 loader; no API cases ran. Failed receipts retained. Exact frozen image reused, no native rebuild or policy change.')
        else:
            run('E01-freeze', [sys.executable, str(HERE / 'freeze_release.py'),
                              '--runtime-image', review['runtime_image_id']])
        release = json.loads((CFG / 'production_image.json').read_text())
        state.update(image_id=release['image_id'], policy_sha256=release['policy_sha256'])
        save()
        operations_name = 'E01-operations-final' + suffix
        operations_argv = [sys.executable, '-u', str(HERE / 'run_operations.py'),
                                   '--image', release['image_id'], '--out',
                                   str(ROOT / operations_name)]
        if args.c16_review_sha256:
            operations_argv += ['--c16-review-sha256', args.c16_review_sha256]
        run(operations_name, operations_argv)
        operations = json.loads((ROOT / operations_name / 'operations.json').read_text())
        assert operations['status'] == 'PASS_NEW_IMAGE_OPERATIONAL_GATES'
        run('E01-serving-final', [sys.executable, '-u',
                                 str(REPO / 'scripts/run-exl3-review-serving.py'),
                                 '--release-dir', str(CFG), '--image-receipt',
                                 str(ROOT / 'E01-final-metadata/image.json'),
                                 '--fixture-root', str(REPO.parent / 'intel-b70-qwen38-vllm/benchmark-results/meaningful-full-profile/run-20260923-201101-w0.00'),
                                 '--out', str(ROOT / 'E01-serving-final')])
        state['status'] = 'COMPLETE_FINAL_OPERATIONS_AND_FROZEN_SERVING_REVIEW_REQUIRED'
    except BaseException as error:
        state.update(status='FAILED', error=repr(error))
        raise
    finally:
        state['finished_unix'] = time.time()
        save()


if __name__ == '__main__':
    main()

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
    out = ROOT / 'E01-final-controller.json'
    assert not out.exists()
    state = dict(status='RUNNING', started_unix=time.time(),
                 serving_review_sha256=args.serving_review_sha256,
                 runtime_image_id=review['runtime_image_id'], commands=[],
                 sources_sha256={str(p.relative_to(REPO)): sha(p) for p in
                                 [Path(__file__), HERE / 'freeze_release.py',
                                  HERE / 'run_operations.py',
                                  REPO / 'scripts/run-exl3-review-serving.py',
                                  REPO / 'scripts/current-profile-benchmark.py']})

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
        run('E01-freeze', [sys.executable, str(HERE / 'freeze_release.py'),
                          '--runtime-image', review['runtime_image_id']])
        release = json.loads((CFG / 'production_image.json').read_text())
        state.update(image_id=release['image_id'], policy_sha256=release['policy_sha256'])
        save()
        run('E01-operations-final', [sys.executable, '-u', str(HERE / 'run_operations.py'),
                                   '--image', release['image_id'], '--out',
                                   str(ROOT / 'E01-operations-final')])
        operations = json.loads((ROOT / 'E01-operations-final/operations.json').read_text())
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

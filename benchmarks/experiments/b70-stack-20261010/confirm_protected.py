#!/usr/bin/env python3
"""Resolve the protected C4 uncertainty with two more unchanged paired blocks."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = REPO / 'benchmark-results/b70-stack-20261010'
PUBLIC = REPO / 'benchmarks/results/b70-stack-20261010'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, required=True)
    args = parser.parse_args()
    out = ROOT / 'E01-additional-protected-controller.json'
    assert not out.exists()
    state = dict(status='WAIT_PREVIOUS_RESTORED_CONTROLLER', previous_pid=args.wait_pid,
                 rationale='Protected source-C4 warm aggregate varies across the initial additional pairs; resolve the remaining concern with five protected independent blocks, retaining all earlier data.',
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 started_unix=time.time())

    def save():
        out.write_text(json.dumps(state, indent=2) + '\n')

    save()
    print('WAIT_PREVIOUS_CONTROLLER', flush=True)
    monitor = None
    try:
        while True:
            try:
                os.kill(args.wait_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(15)
        previous = json.loads((ROOT / 'E01-confirmation-protected/campaign.json').read_text())
        assert previous['status'] == 'COMPLETE_PAIRED_SCREEN_REVIEW_REQUIRED'
        assert previous['restoration']
        state['status'] = 'RUNNING_ADDITIONAL_UNCHANGED_PROTECTED_PANEL'
        save()
        screen = ROOT / 'E01-protected-extended'
        worker = subprocess.Popen([sys.executable, '-u', str(HERE / 'serving_screen.py'),
            '--image', previous['candidate'], '--onednn-receipt',
            str(ROOT / 'E01-oneDNN-build/library-receipt.json'), '--out', str(screen),
            '--blocks', '2', '--protected'], cwd=REPO)
        while not screen.exists() and worker.poll() is None:
            time.sleep(.25)
        if worker.poll() is None:
            monitor = subprocess.Popen([sys.executable, str(HERE / 'monitor_comparability.py'),
                '--controller-pid', str(worker.pid), '--out', str(screen / 'comparability.jsonl')], cwd=REPO)
        code = worker.wait()
        if monitor is not None:
            monitor.wait(timeout=45)
        assert code == 0
        subprocess.run([sys.executable, str(HERE / 'analyze_screen.py'), '--root',
            str(ROOT / 'E01-serving-screen'), '--additional',
            str(ROOT / 'E01-confirmation-protected'), '--additional', str(screen),
            '--out', str(PUBLIC / 'E01-confirmation-eight-blocks.json')], check=True, cwd=REPO)
        state['status'] = 'COMPLETE_EIGHT_CORE_FIVE_PROTECTED_BLOCKS_REVIEW_REQUIRED'
        print(state['status'], flush=True)
    except BaseException as error:
        state.update(status='FAILED', error=repr(error))
        raise
    finally:
        state['finished_unix'] = time.time()
        save()


if __name__ == '__main__':
    main()

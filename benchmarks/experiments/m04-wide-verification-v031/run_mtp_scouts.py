#!/usr/bin/env python3
"""Bounded MTP5..8 scout on matching 32K requests; each child restores serving."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--image', default='local/b70-m04-wide:v031-cache-fixed')
args = parser.parse_args()
root = REPO / 'benchmark-results/m04-wide-verification-v031-20261009' / args.name
root.mkdir(parents=True, exist_ok=False)
(root / 'executing-controller.py').write_text(Path(__file__).read_text())
controller = REPO / 'benchmarks/experiments/dflash2-serving-v031/run_serving.py'
digest = hashlib.sha256(controller.read_bytes()).hexdigest()
state = dict(status='RUNNING_MTP5_TO_8_32K_SCOUT_ONLY',
             image=args.image, started_unix=time.time(), arms=[],
             baseline_regenerated=False, child_controller_sha256=digest)
def save():
    (root / 'campaign.json').write_text(json.dumps(state, indent=2) + '\n')
save()
try:
    for depth in (5, 6, 7, 8):
        assert hashlib.sha256(controller.read_bytes()).hexdigest() == digest
        name = args.name + '-k' + str(depth)
        command = [sys.executable, str(controller), '--name', name,
                   '--method', 'mtp', '--depth', str(depth),
                   '--contexts', '262144,192000', '--benchmark',
                   '--scenario-names', 'phase-32k-c1', '--image', args.image]
        arm = dict(depth=depth, command=command, status='RUNNING')
        state['arms'].append(arm); save()
        print('START_MTP_SCOUT', depth, flush=True)
        with (root / f'k{depth}-controller.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        path = REPO / 'benchmark-results/dflash2-serving-v031-20261009' / name / 'campaign.json'
        campaign = json.loads(path.read_text())
        assert campaign['restoration']['models_endpoint_healthy']
        assert campaign['restoration']['image_id'] == campaign['production_image']
        arm.update(status=campaign['status'], campaign_path=str(path.relative_to(REPO)),
                   campaign_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        save()
        assert result.returncode == 0, campaign.get('error')
        print('PASS_MTP_SCOUT', depth, campaign['active_context_limit'], flush=True)
    compare = [sys.executable, str(controller.parent / 'compare.py')]
    for arm in state['arms']:
        compare += ['--campaign', str(REPO / arm['campaign_path'])]
    compare += ['--out', str(root / 'comparison.json')]
    with (root / 'comparison.log').open('w') as log:
        subprocess.run(compare, stdout=log, stderr=subprocess.STDOUT, check=True)
    state['status'] = 'PASS_MTP5_TO_8_32K_SCOUT_ONLY'
except BaseException as error:
    state.update(status='FAILED', error=repr(error))
    raise
finally:
    state['finished_unix'] = time.time(); save()

#!/usr/bin/env python3
"""Exclusive new-kernel micro gates, with guaranteed qualified restoration."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).parent
sys.path.insert(0, str(REPO / 'scripts'))
from exl3_candidate_worker import profile

spec = importlib.util.spec_from_file_location('restore', REPO / 'scripts/run-exl3-review-micro.py')
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)
parser = argparse.ArgumentParser()
parser.add_argument('--image', default='local/b70-m04-wide:v031-probe')
parser.add_argument('--name', default='micro-default')
args = parser.parse_args()
root = REPO / 'benchmark-results/m04-wide-verification-v031-20261009' / args.name
assert not root.exists()
root.mkdir(parents=True)
production = json.loads((REPO / 'config/experiments/vllm-031-upgrade/production_image.json').read_text())['image_id']
image = subprocess.check_output(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
config = profile(image)
sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.py')}
state = dict(status='RUNNING', image=image, production_image=production,
             started_unix=time.time(), sources=sources)
def save():
    (root / 'campaign.json').write_text(json.dumps(state, indent=2) + '\n')

with (REPO.parent / 'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save()
    try:
        subprocess.run(['systemctl', '--user', 'stop', 'b70-qwen38-vllm.service'], check=True, timeout=90)
        assert subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).splitlines() == ['b70-chatui']
        command = ['docker', 'run', '--rm', '--name', 'b70-m04-wide-micro', '--network', 'none',
                   '--device', '/dev/dri', '--shm-size', '4g', '-v', str(SOURCE) + ':/experiment:ro',
                   '-v', str(root) + ':/results']
        for key, value in {**config['env'], 'ZE_AFFINITY_MASK': '0', 'OMP_NUM_THREADS': '4',
                           'PYTHONPATH': '/experiment:/opt/b70-runtime'}.items():
            command += ['-e', key + '=' + value]
        command += ['--entrypoint', 'python', image, '-u', '/experiment/probe_xpu.py',
                    '--library', '/opt/exl3xpu/m04/m04.so', '--out', '/results/probe']
        state['command'] = command
        save()
        with (root / 'probe.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        block9 = command[:-6] + ['-u', '/experiment/test_block9.py', '--out', '/results/block9.json']
        state['block9_command'] = block9
        save()
        with (root / 'block9.log').open('w') as log:
            subprocess.run(block9, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        assert json.loads((root / 'block9.json').read_text())['status'] == 'PASS_NEW_BLOCK9_FP16_PRIMITIVES'
        result = json.loads((root / 'probe/result.json').read_text())
        assert result['status'] == 'COMPLETE_MICRO_GATES_ONLY_SERVING_UNQUALIFIED'
        assert result['library_sha256'] == config['env']['EXL3_M04_LIBRARY_SHA256']
        assert all(hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest for name, digest in sources.items())
        state.update(status='PASS_NEW_M04_MICRO_GATES_NOT_SERVING_QUALIFICATION',
                     result_sha256=hashlib.sha256((root / 'probe/result.json').read_bytes()).hexdigest())
    except BaseException as error:
        state.update(status='FAILED', error=repr(error))
        raise
    finally:
        subprocess.run(['docker', 'rm', '-f', 'b70-m04-wide-micro'], capture_output=True)
        subprocess.run(['systemctl', '--user', 'start', 'b70-qwen38-vllm.service'], check=True, timeout=90)
        state['restoration'] = restore.wait_for_qualified_service(production, timeout=900)
        state['finished_unix'] = time.time()
        save()

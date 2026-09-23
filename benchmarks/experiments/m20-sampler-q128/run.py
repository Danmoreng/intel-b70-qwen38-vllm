#!/usr/bin/env python3
"""Run short operator probes under the GPU lock, restoring production."""

import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location('m20_diag', REPO / 'scripts/run-diagnostics.py')
diag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diag
spec.loader.exec_module(diag)
IMAGE = 'sha256:648132c9b9da4bb244d7304b956c1a9bb825be640a92755a5ffe32f2bbd679b4'


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f'usage: {sys.argv[0]} RUN_DIR')
    run = Path(sys.argv[1]).resolve()
    run.mkdir(parents=True, exist_ok=False)
    lock = Path('/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    production = diag.production_inspect()
    if production['Image'] != IMAGE:
        raise RuntimeError(f'unexpected production image: {production["Image"]}')
    diag.ensure_idle(diag.PRODUCTION_URL)
    if int(diag.POWER_CAP.read_text()) != 180000000:
        raise RuntimeError('power cap changed')
    (run / 'manifest.json').write_text(json.dumps({
        'image': IMAGE, 'probes': ['sampler_probe.py', 'q128_profile.py'],
        'production_config': production['Args'],
    }, indent=2) + '\n')
    stopped = False
    failures = []
    try:
        stopped = True
        subprocess.run(['systemctl', '--user', 'stop', 'qwen38.service'],
                       check=True, timeout=120)
        for name in ('sampler_probe.py', 'q128_profile.py'):
            command = ['docker', 'run', '--rm', '--device', '/dev/dri',
                       '--group-add', '989', '--shm-size', '4g',
                       '-v', f'{HERE}:/probe:ro', '-v', f'{run}:/evidence',
                       '--entrypoint', 'python', IMAGE, f'/probe/{name}']
            print(f'START {name}', flush=True)
            begin = time.monotonic()
            with (run / f'{name}.log').open('w') as log:
                try:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=600)
                    status = 'PASS'
                except Exception as exc:
                    status = f'FAIL: {type(exc).__name__}: {exc}'
            print(f'{name}: {status} in {time.monotonic()-begin:.1f}s', flush=True)
            if status != 'PASS':
                print((run / f'{name}.log').read_text()[-2500:], flush=True)
                failures.append(f'{name}: {status}')
    finally:
        if stopped:
            subprocess.run(['systemctl', '--user', 'start', 'qwen38.service'],
                           check=True, timeout=900)
            diag.ensure_idle(diag.PRODUCTION_URL)
            restored = diag.production_inspect()
            if restored['Image'] != IMAGE or int(diag.POWER_CAP.read_text()) != 180000000:
                raise RuntimeError('production restoration failed')
            (run / 'production-restored.json').write_text(json.dumps({
                'image': restored['Image'], 'power_cap_uw': int(diag.POWER_CAP.read_text()),
                'active': True}, indent=2) + '\n')
            print('PRODUCTION RESTORED', flush=True)
    if failures:
        raise RuntimeError('; '.join(failures))


if __name__ == '__main__':
    main()

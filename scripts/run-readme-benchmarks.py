#!/usr/bin/env python3
"""Run the complete public workloads on the frozen permanent-service image."""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
SERVICE = 'b70-qwen38-vllm.service'
CONTAINER = 'b70-qwen38-vllm'
BASE = 'http://127.0.0.1:8081'


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity():
    item = json.loads(subprocess.check_output(['docker', 'inspect', CONTAINER], text=True))[0]
    return {'image_id': item['Image'], 'image_tag': item['Config']['Image'],
            'container_id': item['Id'], 'command': item['Config']['Cmd'],
            'policy_sha256': item['Config']['Labels'].get('org.local.b70.policy.sha256')}


def healthy(release):
    try:
        with urllib.request.urlopen(BASE + '/health', timeout=3) as response:
            return response.status == 200 and identity()['image_id'] == release['image_id']
    except (OSError, subprocess.CalledProcessError):
        return False


def ensure_serving(release, restart=False):
    if restart or not healthy(release):
        subprocess.run(['systemctl', '--user', 'restart', SERVICE], check=True, timeout=120)
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if healthy(release):
            return identity()
        time.sleep(5)
    raise RuntimeError('permanent service did not become healthy on the frozen image')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--fixture-root', type=Path)
    parser.add_argument('--ensure-serving', action='store_true')
    args = parser.parse_args()
    release = json.loads((REPO / 'config/production_image.json').read_text())
    if args.ensure_serving:
        print(json.dumps(ensure_serving(release)), flush=True)
        return
    if args.output_root is None or args.fixture_root is None:
        parser.error('--output-root and --fixture-root are required for a benchmark')
    root = args.output_root.resolve()
    if root.exists():
        raise RuntimeError('use a fresh output directory')
    lock_path = REPO.parent / 'Local-AI-B70/qwen38/context-benchmark/run.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(parents=True)
        started = datetime.datetime.now(ZoneInfo('Europe/Berlin')).isoformat()
        phase = 'preflight'
        save(root / 'state.json', {'status': 'running', 'phase': phase, 'started_at': started})
        try:
            policy = digest(REPO / 'config/production_policy.json')
            if policy != release['policy_sha256']:
                raise RuntimeError('release policy differs')
            image = subprocess.check_output(['docker', 'image', 'inspect', release['image_tag'],
                                             '--format', '{{.Id}}'], text=True).strip()
            if image != release['image_id']:
                raise RuntimeError('production tag differs from frozen image')
            manifest = json.loads((REPO / 'config/frozen_fixture_manifest.json').read_text())
            for file, expected in manifest['files'].items():
                if digest(args.fixture_root / file) != expected:
                    raise RuntimeError('fixture hash differs: ' + file)
            fixtures = {}
            scenarios = json.loads((REPO / 'benchmarks/current-profile-scenarios.json').read_text())
            for scenario in scenarios:
                for repeat in range(1, scenario['repeats'] + 1):
                    folder = args.fixture_root / f"{scenario['name']}-r{repeat}"
                    for request in range(1, scenario['concurrency'] + 1):
                        for name in (f'prompt-{request}.txt', f'request-{request}.json'):
                            file = folder / name
                            fixtures[str(file.relative_to(args.fixture_root))] = digest(file)
            save(root / 'fixture-sha256.json', fixtures)
            source_files = ['scripts/current-profile-benchmark.py', 'scripts/decode_overlap.py',
                            'scripts/run-coding-benchmark.py', 'scripts/run-readme-benchmarks.py',
                            'scripts/run-server.sh', 'config/production_image.json',
                            'benchmarks/current-profile-scenarios.json', 'config/frozen_fixture_manifest.json']
            save(root / 'provenance.json', {'started_at': started, 'release': release,
                 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                 'source_sha256': {file: digest(REPO / file) for file in source_files},
                 'fixture_root': str(args.fixture_root.resolve()), 'fixture_files': len(fixtures),
                 'scope': 'complete 20-scenario serving profile and QueueKit coding fixture v2; no old-image rerun'})
            print('Starting fresh permanent-service worker on', image, flush=True)
            live = ensure_serving(release, restart=True)
            if live['policy_sha256'] != policy:
                raise RuntimeError('live policy differs')
            save(root / 'production-identity.json', live)
            phase = 'source-review'
            save(root / 'state.json', {'status': 'running', 'phase': phase, 'started_at': started})
            subprocess.run([sys.executable, str(REPO / 'scripts/current-profile-benchmark.py'),
                '--expected-max-num-seqs', str(json.loads((REPO / 'config/production_policy.json').read_text())['serving']['max_num_seqs']),
                '--container', CONTAINER, '--fixture-root', str(args.fixture_root.resolve()),
                '--legacy-prefix-namespace', '--output-root', str(root / 'source-review'), '--execute'],
                cwd=REPO, check=True)
            if identity() != live:
                raise RuntimeError('serving identity changed during benchmark')
            phase = 'coding-agent-v2'
            save(root / 'state.json', {'status': 'running', 'phase': phase, 'started_at': started})
            subprocess.run([sys.executable, str(REPO / 'scripts/run-coding-benchmark.py'),
                '--container', CONTAINER, '--output-root', str(root / 'coding-agent-v2')], cwd=REPO, check=True)
            if identity() != live:
                raise RuntimeError('serving identity changed during coding benchmark')
            save(root / 'state.json', {'status': 'complete', 'phase': phase, 'started_at': started,
                 'finished_at': datetime.datetime.now(ZoneInfo('Europe/Berlin')).isoformat()})
        except BaseException as error:
            save(root / 'state.json', {'status': 'failed', 'phase': phase, 'started_at': started,
                                     'error': str(error)})
            raise
        finally:
            save(root / 'production-serving.json', ensure_serving(release))


if __name__ == '__main__':
    main()

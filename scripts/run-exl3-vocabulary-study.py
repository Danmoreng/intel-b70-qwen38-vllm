#!/usr/bin/env python3
"""Small fresh pruned/full draft-vocabulary comparison at fixed MTP3 depth."""
import argparse
import fcntl
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
HELPERS = REPO / 'benchmarks/experiments/quantization-reference'
SPEC = importlib.util.spec_from_file_location('mtp_study', REPO / 'scripts/run-exl3-mtp-study.py')
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)
NAME = M.NAME


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True); parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True); parser.add_argument('--long-campaign', type=Path, required=True)
    parser.add_argument('--generation-campaign', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); root = args.out.resolve(); assert not root.exists()
    encoded = args.panel.read_bytes(); raw = gzip.decompress(encoded); panel = json.loads(raw)
    manifest = json.loads(args.manifest.read_text())
    assert hashlib.sha256(raw).hexdigest() == manifest['panel_raw_sha256'] and sha(args.panel) == manifest['panel_gzip_sha256']
    cases = [(w, c, cache) for w in panel['windows'] for c in [1, 4] for cache in ['cold', 'warm']
             if w['context_tokens'] == 4096 or (w['name'] == 'code-49152' and c == 4)
             or (w['name'] == 'code-102752' and c == 1)]
    assert len(cases) == 12
    image = subprocess.check_output(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    lockpath = REPO.parent / 'Local-AI-B70/qwen38/context-benchmark/run.lock'
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        quality = json.loads((args.long_campaign / 'candidate-campaign.json').read_text())
        generation = json.loads((args.generation_campaign / 'campaign.json').read_text())
        assert quality['status'] == generation['status'] == 'COMPLETE'
        assert quality['image_id'] == generation['image_id'] == image
        assert generation['panel_sha256'] == manifest['panel_raw_sha256']
        settings = json.loads((REPO / 'config/experiments/exl3-migration/target-upstream-expanded.json').read_text())
        assert all(quality['engine_config'].get(k) == v for k, v in settings.items())
        root.mkdir(); state = {'status': 'RUNNING', 'image_id': image, 'panel_sha256': manifest['panel_raw_sha256'],
            'started_unix': time.time(), 'depth': 3, 'arms': [], 'planned_waves': 24,
            'generation_campaign_sha256': sha(args.generation_campaign / 'campaign.json'),
            'long_quality_sha256': sha(args.long_campaign / 'candidate-campaign.json'),
            'selection_note': 'MTP3 is selected as the EXL3 target-profile default for subsequent optimization: long C1 shows only~1% MTP4 gain, prose/C4 regresses about8%, and warm-cache residency differs. No further depth matrix is planned; production release remains gated separately.',
            'comparison_design': 'Fresh pruned then full, fixed depth3/profile/token budget; single-pass screen, no ABBA/order-drift confidence.',
            'source_sha256': {str(p.relative_to(REPO)): sha(p) for p in [Path(__file__), REPO / 'scripts/run-exl3-mtp-study.py', HELPERS / 'head_ownership_audit.py']}}

        def persist(): M.save(root / 'campaign.json', state)
        def interrupted(signum, frame): raise InterruptedError(f'Signal {signum}')
        signal.signal(signal.SIGINT, interrupted); signal.signal(signal.SIGTERM, interrupted); persist()
        try:
            subprocess.run(['systemctl', '--user', 'stop', 'b70-qwen38-vllm.service'], check=True, timeout=90); M.R.cap()
            checkpoint = Path.home() / '.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw'
            for index, vocab in enumerate(['pruned', 'full']):
                phase = root / vocab; phase.mkdir()
                command = ['docker', 'run', '-d', '--name', NAME, '--device', '/dev/dri',
                    '-v', '/dev/dri/by-path:/dev/dri/by-path:ro', '--shm-size', '8g', '-p', '127.0.0.1:8082:8000',
                    '-e', 'HF_HUB_OFFLINE=1', '-e', 'PYTHONPATH=/opt/b70-runtime:/opt/head-audit',
                    '-v', str(checkpoint) + ':/models/checkpoint:ro', '-v', str(REPO / 'runtime') + ':/opt/b70-runtime:ro',
                    '-v', str(HELPERS) + ':/opt/head-audit:ro', '-v', str(root) + ':/results']
                cache = Path.home() / '.cache/exl3xpu/migration-vocabulary' / image.removeprefix('sha256:') / vocab
                for key, target in [('vllm', '/root/.cache/vllm'), ('triton', '/root/.triton/cache'), ('neo_compiler_cache', '/root/.cache/neo_compiler_cache')]:
                    (cache / key).mkdir(parents=True, exist_ok=True); command += ['-v', str(cache / key) + ':' + target]
                command += [image, 'models/qwen3.8-27b-exl3-4.00bpw/migration-target-200704-c4.yaml',
                            '--gpu', '0', '--port', '8000', '--model-path', '/models/checkpoint']
                applied = {**settings, 'speculative_config': {'method': 'mtp', 'num_speculative_tokens': 3},
                           'worker_extension_cls': 'head_ownership_audit.HeadOwnershipExtension'}
                for key, value in applied.items(): command += ['--set', 'vllm.' + key + '=' + json.dumps(value, separators=(',', ':'))]
                command += ['--set', f'env.EXL3_HEAD_OWNERSHIP_REPORT=/results/{vocab}-head-ownership.json',
                            '--set', f'env.EXL3_LOADER_REPORT_DIR=/results/loader-{vocab}']
                if vocab == 'full': command += ['--set', 'env.EXL3_DRAFT_VOCAB=null']
                arm = {'vocabulary': vocab, 'settings': applied, 'command': command, 'waves': [], 'status': 'RUNNING'}
                state['arms'].append(arm); persist()
                try:
                    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
                    deadline = time.monotonic() + 900
                    while True:
                        try: M.R.http(M.BASE, '/v1/models', timeout=3); break
                        except OSError:
                            item = json.loads(subprocess.check_output(['docker', 'inspect', NAME], text=True))[0]
                            assert item['State']['Running'], 'Vocabulary engine exited during startup'
                            if time.monotonic() > deadline: raise TimeoutError('Vocabulary startup timeout')
                            time.sleep(2)
                    identity = M.R.identity(NAME); assert identity['image_id'] == image; arm['identity'] = identity
                    report = root / f'{vocab}-head-ownership.json'; heads = json.loads(report.read_text())
                    assert all(h['full_rows'] == 248320 and h['bits'] == 6 for h in heads['heads'])
                    assert all(h['pruned_rows'] == (65536 if vocab == 'pruned' else None) for h in heads['heads'])
                    arm['head_ownership'] = heads; arm['head_ownership_sha256'] = sha(report); persist()
                    short = next(w for w in panel['windows'] if w['name'] == 'code-4096')
                    for c in [1, 4]: M.wave(short, c, 'warmup', index, 3, phase, 512)
                    for window, c, mode in cases:
                        print(f'RUN {vocab} {window["name"]} C{c} {mode}', flush=True)
                        arm['waves'].append(M.wave(window, c, mode, index, 3, phase, 512)); persist()
                    arm['status'] = 'COMPLETE'; persist()
                finally:
                    logs = subprocess.run(['docker', 'logs', NAME], capture_output=True, text=True)
                    (phase / 'worker.log').write_text(logs.stdout + logs.stderr)
                    subprocess.run(['docker', 'stop', '-t', '20', NAME], capture_output=True, timeout=60)
                    subprocess.run(['docker', 'rm', '-f', NAME], capture_output=True, timeout=30)
            state['status'] = 'COMPLETE'
        except BaseException as exc:
            state['status'] = 'FAILED'; state['error'] = repr(exc); raise
        finally:
            subprocess.run(['docker', 'rm', '-f', NAME], capture_output=True, timeout=30)
            subprocess.run(['systemctl', '--user', 'stop', 'b70-qwen38-vllm.service'], check=True, timeout=90)
            state['production_left_offline'] = True; state['finished_unix'] = time.time(); persist()


if __name__ == '__main__': main()

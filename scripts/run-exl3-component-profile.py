#!/usr/bin/env python3
"""Collect diagnostic event spans after the uninstrumented MTP serving study."""
import argparse
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / 'benchmarks/experiments/quantization-reference'
NAME = 'b70-exl3-component-profile'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--serving-campaign', type=Path, required=True)
    parser.add_argument('--long-campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contexts', type=int, nargs='+', default=[4096, 49152, 102752])
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists(), 'Fresh diagnostic evidence required'
    encoded = args.panel.read_bytes(); panel = gzip.decompress(encoded)
    manifest = json.loads(args.manifest.read_text())
    assert hashlib.sha256(panel).hexdigest() == manifest['panel_raw_sha256']
    assert sha(args.panel) == manifest['panel_gzip_sha256']
    image = subprocess.check_output(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    lockpath = REPO.parent / 'Local-AI-B70/qwen38/context-benchmark/run.lock'
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        serving = json.loads((args.serving_campaign / 'campaign.json').read_text())
        quality = json.loads((args.long_campaign / 'candidate-campaign.json').read_text())
        assert serving['status'] == quality['status'] == 'COMPLETE'
        assert serving['image_id'] == quality['image_id'] == image
        assert serving['scope'] == 'SERVING_ABBA' and serving['order'] == [3, 4, 4, 3]
        assert serving['panel_sha256'] == manifest['panel_raw_sha256']
        config = quality['engine_config']
        settings = json.loads((REPO / 'config/experiments/exl3-migration/target-upstream-expanded.json').read_text())
        assert all(config.get(key) == value for key, value in settings.items())
        root.mkdir(); (root / 'panel.json.gz').write_bytes(encoded)
        power = list(Path('/sys/bus/pci/devices/0000:03:00.0/hwmon').glob('hwmon*/power1_cap'))
        assert len(power) == 1 and int(power[0].read_text()) == 180000000, 'B70 power cap must remain180W'
        state = {'schema': 1, 'status': 'RUNNING', 'image_id': image,
                 'kind': 'DIAGNOSTIC_NOT_SERVING_THROUGHPUT', 'started_unix': time.time(),
                 'serving_campaign_sha256': sha(args.serving_campaign / 'campaign.json'),
                 'long_campaign_sha256': sha(args.long_campaign / 'candidate-campaign.json'),
                 'panel_sha256': manifest['panel_raw_sha256'], 'contexts': args.contexts,
                 'arms': [], 'power_w': 180, 'production_restored': False,
                 'source_sha256': {str(p.relative_to(REPO)): sha(p) for p in
                    [Path(__file__), SCRIPTS / 'profile_capture.py', SCRIPTS / 'run_component_profile.py']}}

        def persist():
            (root / 'campaign.json').write_text(json.dumps(state, indent=2) + '\n')

        def interrupted(signum, frame):
            raise InterruptedError(f'Signal {signum}')

        signal.signal(signal.SIGINT, interrupted); signal.signal(signal.SIGTERM, interrupted)
        persist()
        try:
            subprocess.run(['systemctl', '--user', 'stop', 'b70-qwen38-vllm.service'], check=True, timeout=90)
            checkpoint = Path.home() / '.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw'
            for depth in [3, 4]:
                arm = {'depth': depth, 'status': 'RUNNING'}; state['arms'].append(arm)
                applied = {**config, 'speculative_config': {'method': 'mtp', 'num_speculative_tokens': depth}}
                cfg = root / f'mtp{depth}-config.json'; cfg.write_text(json.dumps(applied, indent=2) + '\n')
                cache = Path.home() / '.cache/exl3xpu/migration-components' / image.removeprefix('sha256:') / f'mtp{depth}'
                command = ['docker', 'run', '--rm', '--name', NAME, '--network', 'none', '--device', '/dev/dri',
                           '--memory', '12g', '--shm-size', '4g', '-v', '/dev/dri/by-path:/dev/dri/by-path:ro',
                           '-v', str(checkpoint) + ':/exl3:ro', '-v', str(SCRIPTS) + ':/scripts:ro', '-v', str(root) + ':/results']
                for key, target in [('vllm', '/root/.cache/vllm'), ('triton', '/root/.triton/cache'), ('neo_compiler_cache', '/root/.cache/neo_compiler_cache')]:
                    (cache / key).mkdir(parents=True, exist_ok=True); command += ['-v', str(cache / key) + ':' + target]
                env = {'HF_HUB_OFFLINE': '1', 'PYTHONPATH': '/scripts', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
                       'ZE_FLAT_DEVICE_HIERARCHY': 'COMPOSITE', 'ZE_AFFINITY_MASK': '0', 'OMP_NUM_THREADS': '4',
                       'EXL3_TARGET_RUNTIME': 'vllm030', 'EXL3_INT8_PREFILL': '1', 'EXL3_ONEDNN_ATTN': '0',
                       'EXL3_DRAFT_VOCAB': '/opt/exl3xpu/models/qwen3.8-27b-exl3-4.00bpw/draft_vocab.json',
                       'EXL3_LOADER_REPORT_DIR': f'/results/loader-mtp{depth}', 'VLLM_XPU_ENABLE_XPU_GRAPH': '1',
                       'B70_ONEDNN_PREFILL': '0', 'B70_ONEDNN_MIXED_ROUTE': '0', 'B70_GPTQ_W4A8_PREFILL': '0'}
                for key, value in env.items(): command += ['-e', key + '=' + value]
                command += ['--entrypoint', 'python', image, '-u', '/scripts/run_component_profile.py',
                            '--model', '/exl3', '--panel', '/results/panel.json.gz', '--engine-config', '/results/' + cfg.name,
                            '--out', f'/results/mtp{depth}', '--contexts', *map(str, args.contexts)]
                arm['command'] = command; arm['engine_config'] = applied; persist()
                with (root / f'mtp{depth}.log').open('w') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                summary = root / f'mtp{depth}/summary.json'
                assert json.loads(summary.read_text())['status'] == 'COMPLETE'
                arm['status'] = 'COMPLETE'; arm['summary_sha256'] = sha(summary); persist()
            state['status'] = 'COMPLETE'
        except BaseException as exc:
            state['status'] = 'FAILED'; state['error'] = repr(exc); raise
        finally:
            subprocess.run(['docker', 'rm', '-f', NAME], capture_output=True, timeout=30)
            subprocess.run(['systemctl', '--user', 'stop', 'b70-qwen38-vllm.service'], check=True, timeout=90)
            state['production_left_offline'] = True; state['finished_unix'] = time.time(); persist()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Queue one eager kernel trace after the bounded graph component campaign."""
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
HELPERS = REPO / 'benchmarks/experiments/quantization-reference'
NAME = 'b70-exl3-eager-trace'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--components', type=Path, required=True)
    parser.add_argument('--long-campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); root = args.out.resolve()
    assert not root.exists()
    encoded = args.panel.read_bytes(); raw = gzip.decompress(encoded)
    manifest = json.loads(args.manifest.read_text())
    assert hashlib.sha256(raw).hexdigest() == manifest['panel_raw_sha256']
    assert sha(args.panel) == manifest['panel_gzip_sha256']
    image = subprocess.check_output(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    lockpath = REPO.parent / 'Local-AI-B70/qwen38/context-benchmark/run.lock'
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        components = json.loads((args.components / 'campaign.json').read_text())
        quality = json.loads((args.long_campaign / 'candidate-campaign.json').read_text())
        assert components['status'] == quality['status'] == 'COMPLETE'
        assert components['image_id'] == quality['image_id'] == image
        assert components['compact'] and all(a['status'] == 'COMPLETE' for a in components['arms'])
        assert components['panel_sha256'] == manifest['panel_raw_sha256']
        config = {**quality['engine_config'], 'dtype': 'float16', 'seed': 20261001,
                  'enforce_eager': True, 'compilation_config': {'cudagraph_mode': 'NONE'},
                  'speculative_config': {'method': 'mtp', 'num_speculative_tokens': 3}}
        root.mkdir(); (root / 'panel.json.gz').write_bytes(encoded)
        (root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        state = {'status': 'RUNNING', 'image_id': image, 'started_unix': time.time(),
                 'kind': 'BOUNDED_EAGER_DIAGNOSTIC_NOT_SERVING_BENCHMARK',
                 'components_sha256': sha(args.components / 'campaign.json'),
                 'long_quality_sha256': sha(args.long_campaign / 'candidate-campaign.json'),
                 'panel_sha256': manifest['panel_raw_sha256'], 'engine_config': config,
                 'source_sha256': {str(p.relative_to(REPO)): sha(p) for p in
                     [Path(__file__), HELPERS / 'run_eager_kernel_trace.py', HELPERS / 'profile_capture.py']}}

        def persist():
            (root / 'campaign.json').write_text(json.dumps(state, indent=2) + '\n')

        def interrupted(signum, frame):
            raise InterruptedError(f'Signal {signum}')

        signal.signal(signal.SIGINT, interrupted); signal.signal(signal.SIGTERM, interrupted)
        persist()
        try:
            subprocess.run(['systemctl', '--user', 'stop', 'b70-qwen38-vllm.service'], check=True, timeout=90)
            cap = list(Path('/sys/bus/pci/devices/0000:03:00.0/hwmon').glob('hwmon*/power1_cap'))
            assert len(cap) == 1 and int(cap[0].read_text()) == 180000000
            checkpoint = Path.home() / '.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw'
            cache = Path.home() / '.cache/exl3xpu/migration-eager-trace' / image.removeprefix('sha256:')
            command = ['docker', 'run', '--rm', '--name', NAME, '--network', 'none', '--device', '/dev/dri',
                       '--memory', '12g', '--shm-size', '4g', '-v', '/dev/dri/by-path:/dev/dri/by-path:ro',
                       '-v', str(checkpoint) + ':/exl3:ro', '-v', str(HELPERS) + ':/scripts:ro', '-v', str(root) + ':/results']
            for key, target in [('vllm', '/root/.cache/vllm'), ('triton', '/root/.triton/cache'), ('neo_compiler_cache', '/root/.cache/neo_compiler_cache')]:
                (cache / key).mkdir(parents=True, exist_ok=True); command += ['-v', str(cache / key) + ':' + target]
            env = {'HF_HUB_OFFLINE': '1', 'PYTHONPATH': '/scripts', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
                   'ZE_FLAT_DEVICE_HIERARCHY': 'COMPOSITE', 'ZE_AFFINITY_MASK': '0', 'OMP_NUM_THREADS': '4',
                   'EXL3_TARGET_RUNTIME': 'vllm030', 'EXL3_INT8_PREFILL': '1', 'EXL3_ONEDNN_ATTN': '0',
                   'EXL3_DRAFT_VOCAB': '/opt/exl3xpu/models/qwen3.8-27b-exl3-4.00bpw/draft_vocab.json',
                   'EXL3_LOADER_REPORT_DIR': '/results/loader', 'VLLM_XPU_ENABLE_XPU_GRAPH': '0',
                   'B70_ONEDNN_PREFILL': '0', 'B70_ONEDNN_MIXED_ROUTE': '0', 'B70_GPTQ_W4A8_PREFILL': '0'}
            for key, value in env.items(): command += ['-e', key + '=' + value]
            command += ['--entrypoint', 'python', image, '-u', '/scripts/run_eager_kernel_trace.py',
                        '--model', '/exl3', '--panel', '/results/panel.json.gz', '--engine-config', '/results/config.json',
                        '--out', '/results/capture']
            state['command'] = command; persist()
            with (root / 'worker.log').open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            summary = root / 'capture/summary.json'
            assert json.loads(summary.read_text())['status'] == 'COMPLETE'
            state['status'] = 'COMPLETE'; state['summary_sha256'] = sha(summary)
        except BaseException as exc:
            state['status'] = 'FAILED'; state['error'] = repr(exc); raise
        finally:
            subprocess.run(['docker', 'rm', '-f', NAME], capture_output=True, timeout=30)
            subprocess.run(['systemctl', '--user', 'stop', 'b70-qwen38-vllm.service'], check=True, timeout=90)
            state['production_left_offline'] = True; state['finished_unix'] = time.time(); persist()


if __name__ == '__main__':
    main()

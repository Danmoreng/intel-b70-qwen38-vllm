#!/usr/bin/env python3
"""Launch isolated RAM-cache candidate; never changes the qualified service or aliases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from exl3_candidate_worker import MODEL, REPO, profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-gib', type=int, default=12)
    p.add_argument('--port', type=int, default=8082)
    p.add_argument('--name', default='b70-kv-ram-candidate')
    p.add_argument('--gpu-utilization', type=float, default=.965)
    p.add_argument('--context', type=int, default=262144)
    p.add_argument('--watermark', type=float, default=.05)
    p.add_argument('--cache-policy', choices=['lru','arc'], default='arc')
    p.add_argument('--native-admission', action='store_true')
    p.add_argument('--no-offload', action='store_true')
    p.add_argument('--test-endpoints', action='store_true', help='Expose cache-reset API on localhost test server')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not 1 <= a.cpu_gib <= 12 or not 1024 <= a.port <= 65535 or a.port == 8081:
        p.error('RAM must be 1–12 GiB and candidate port must differ from production 8081')
    if not .5 <= a.gpu_utilization <= .965 or not 4096 <= a.context <= 262144 or not 0 <= a.watermark < .25:
        p.error('Candidate resource limits outside safe bounds')
    if a.name == 'b70-qwen38-vllm' or not a.name.startswith('b70-kv-'):
        p.error('Candidate must use its own b70-kv-* container name')
    image = json.loads((REPO/'config/production_image.json').read_text())['image_id']
    cfg = profile(image)
    argv = cfg['argv'][1:]
    def setting(flag, value):
        nonlocal argv
        if flag in argv:
            i = argv.index(flag); del argv[i:i+2]
        argv += [flag, str(value)]
    setting('--gpu-memory-utilization', a.gpu_utilization)
    setting('--max-model-len', a.context)
    setting('--watermark', a.watermark)
    if not a.native_admission:
        setting('--scheduler-cls', 'kv_admission.CompletionReservedScheduler')
    if not a.no_offload:
        setting('--kv-transfer-config', json.dumps({
            'kv_connector': 'OffloadingConnector', 'kv_role': 'kv_both',
            'kv_connector_extra_config': {'cpu_bytes_to_use': a.cpu_gib*1024**3,
                'spec_name': 'BoundedXPUOffloadingSpec', 'spec_module_path': 'xpu_cpu_offload',
                'blocks_per_chunk': 1, 'eviction_policy': a.cache_policy, 'store_threshold': 0}}))
    cache = Path.home()/'.cache/b70-kv-ram-candidate'
    result = REPO/'benchmark-results/kv-ram-candidate'
    result.mkdir(parents=True, exist_ok=True)
    env = {**cfg['env'], 'HF_HUB_OFFLINE': '1', 'PYTHONPATH': '/opt/b70-runtime',
           'ZE_AFFINITY_MASK': '0', 'EXL3_LOADER_REPORT_DIR': '/results/loader'}
    if a.test_endpoints:
        env['VLLM_SERVER_DEV_MODE'] = '1'
    command = ['docker', 'run', '-d', '--name', a.name, '--device', '/dev/dri',
        '--shm-size', f'{a.cpu_gib+2}g', '--memory', '23g', '--memory-swap', '23g',
        '-p', f'127.0.0.1:{a.port}:8000', '-v', f'{MODEL}:/models/checkpoint:ro',
        '-v', f'{REPO / "runtime"}:/opt/b70-runtime:ro', '-v', f'{result}:/results']
    for key, target in [('vllm','/root/.cache/vllm'),('triton','/root/.triton/cache'),('neo_compiler_cache','/root/.cache/neo_compiler_cache')]:
        directory = cache/key; directory.mkdir(parents=True, exist_ok=True)
        command += ['-v', f'{directory}:{target}']
    for key, value in env.items(): command += ['-e', f'{key}={value}']
    command += ['--entrypoint', '/opt/venv/bin/vllm', image, *argv]
    cache_budget = 0 if a.no_offload else a.cpu_gib*1024**3
    receipt = {'command': command, 'cpuCacheBudgetBytes': cache_budget,
        'admission': 'native' if a.native_admission else 'input-and-output-reserved',
        'offload': not a.no_offload, 'baseImage': image,
        'cachePolicy': None if a.no_offload else a.cache_policy,
        'runtimeSha256': {path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (REPO/'runtime').glob('*.py')}}
    if a.dry_run:
        print(json.dumps(receipt, indent=2)); return
    if subprocess.run(['docker','inspect', a.name], capture_output=True).returncode == 0:
        raise SystemExit('Candidate container exists; refusing replacement')
    if subprocess.run(['docker','inspect','b70-qwen38-vllm','--format','{{.State.Running}}'],capture_output=True,text=True).stdout.strip()=='true':
        raise SystemExit('Production model still running; GPU test refused')
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    if available < cache_budget + 7*1024**3:
        raise SystemExit('Insufficient available RAM for cache plus 7 GiB runtime/headroom')
    (result/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    subprocess.run(command,check=True)


if __name__ == '__main__': main()

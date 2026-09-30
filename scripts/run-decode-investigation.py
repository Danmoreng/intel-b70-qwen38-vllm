#!/usr/bin/env python3
"""Exclusive offline B70 decode controls, with verified service restoration."""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
SERVICE = 'b70-qwen38-vllm.service'
PRODUCTION = 'b70-qwen38-vllm'
CONTAINER = 'b70-decode-investigation'
BASE = 'http://127.0.0.1:18087'
IMAGES = {
    'old': 'sha256:648132c9b9da4bb244d7304b956c1a9bb825be640a92755a5ffe32f2bbd679b4',
    'current': 'sha256:a42cda993bf6492acc39d23e9382e27a17efca4657bea07d80d0c28228a1623a',
    'bypass': 'sha256:a42cda993bf6492acc39d23e9382e27a17efca4657bea07d80d0c28228a1623a',
    'fixed': 'sha256:89b607479ecf61a251aa23271fefedee90edaa38b91440a5889293b4e72d6035',
    'fused': 'sha256:ed1ebca756abb0e0832d11cd0db026dd7e86df094c6903efe7ae8afbdc290b68',
}
FIXTURE = Path('/home/sebastian/LocalLLM/intel-b70-qwen38-vllm/benchmark-results/meaningful-full-profile/run-20260923-201101-w0.00')
LOCK = Path('/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock')
spec = importlib.util.spec_from_file_location('decode_benchmark', REPO / 'scripts/current-profile-benchmark.py')
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def inspect(name):
    return json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]


def healthy(base):
    try:
        with urllib.request.urlopen(base + '/health', timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def restore(root):
    if not (root / 'production-identity.json').exists():
        return
    subprocess.run(['docker', 'stop', '-t', '30', CONTAINER],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
    subprocess.run(['systemctl', '--user', 'start', SERVICE], check=True, timeout=60)
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if healthy('http://127.0.0.1:8081'):
            item = inspect(PRODUCTION)
            expected = json.loads((root / 'production-identity.json').read_text())
            if item['Image'] != expected['image_id']:
                raise RuntimeError('restored image identity differs')
            if item['Config']['Cmd'] != expected['command']:
                raise RuntimeError('restored serving arguments differ')
            save(root / 'production-restored.json', {
                'image_id': item['Image'], 'service': SERVICE, 'health': 200,
                'restored_unix': time.time(), 'command_matches': True})
            return
        time.sleep(2)
    raise TimeoutError('production restoration did not become healthy')


def command(item, arm, phase, root, profile=False, capture=False, cycle=False, verified=False, cycle_fixture_root=None):
    image = IMAGES[arm]
    source_root = phase / 'sources'
    source_root.mkdir()
    hashes = {}
    for name in ('b70_decode_trace.py', 'b70_decode_trace.pth', 'b70_linear_capture.py',
                 'b70_linear_capture.pth', 'b70_cycle_replay.py', 'b70_cycle_replay.pth'):
        source = REPO / 'scripts' / name
        shutil.copyfile(source, source_root / name)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    save(phase / 'instrumentation-source-sha256.json', hashes)
    mode = 'trace' if profile else 'plain'
    if arm not in ('old', 'current'):
        mode += '-' + arm
    cache = root.parent / 'compile-caches' / image.removeprefix('sha256:') / mode
    for name in ('vllm', 'triton'):
        (cache / name).mkdir(parents=True, exist_ok=True)
    result = ['docker', 'run', '--rm', '--name', CONTAINER,
              '--device', '/dev/dri', '--group-add', str(Path('/dev/dri/renderD128').stat().st_gid),
              '--shm-size', '8g', '-p', '127.0.0.1:18087:8000',
              '-v', f'{phase}:/evidence', '-v', '/dev/dri:/dev/dri:ro']
    for mount in item['Mounts']:
        dest = mount['Destination']
        if dest in ('/dev/dri', '/root/.cache/vllm', '/root/.triton/cache'):
            continue
        suffix = '' if mount['RW'] else ':ro'
        result += ['-v', f"{mount['Source']}:{dest}{suffix}"]
    result += ['-v', f'{cache / "vllm"}:/root/.cache/vllm',
               '-v', f'{cache / "triton"}:/root/.triton/cache']
    env = dict(row.split('=', 1) for row in item['Config']['Env']
               if row.startswith(('B70_', 'VLLM_', 'ZE_', 'PYTORCH_',
                                  'HF_HUB_OFFLINE=', 'PYTHONPATH=', 'MODEL_')))
    env.update(B70_GPTQ_W4A8_PREFILL='0' if arm == 'old' else '1',
               B70_ONEDNN_PREFILL='0' if arm == 'old' else '1',
               B70_ONEDNN_MIXED_ROUTE='0' if arm == 'old' else '1')
    for key, value in env.items():
        result += ['-e', f'{key}={value}']
    arguments = item['Config']['Cmd'].copy()
    if arm == 'bypass':
        source = (REPO / 'docker/onednn-prefill/b70_attention.py').read_text()
        anchor = 'def flash_attn_varlen_func(**d):\n    q = d["q"]\n'
        if source.count(anchor) != 1:
            raise RuntimeError('unexpected attention adapter source')
        source = source.replace(anchor, anchor +
            '    # Offline intervention: this domain can never use oneDNN.\n'
            '    if q.shape[0] < 256:\n        return fallback(**d)\n')
        override = phase / 'b70_attention.py'
        override.write_text(source)
        save(phase / 'source-overrides.json', {'attention_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'intervention': 'small-query early fallback; native libraries remain loaded'})
        result += ['-v', f'{override}:/opt/venv/lib/python3.12/site-packages/b70_attention.py:ro']
    if profile:
        result += ['-v', f'{source_root / "b70_decode_trace.py"}:/opt/b70-trace/b70_decode_trace.py:ro',
                   '-v', f'{source_root / "b70_decode_trace.pth"}:/opt/venv/lib/python3.12/site-packages/b70_decode_trace.pth:ro',
                   '-e', 'B70_DECODE_TRACE=1', '-e', 'PYTHONPATH=/opt/b70-runtime:/opt/b70-trace']
        arguments += ['--profiler-config', json.dumps({
            'profiler': 'torch', 'torch_profiler_dir': '/evidence/traces',
            'torch_profiler_record_shapes': True, 'torch_profiler_with_stack': False}),
            '--cudagraph-metrics']
    if cycle:
        fixtures = cycle_fixture_root or root / 'cycle-fixtures'
        fixtures.mkdir(exist_ok=True)
        result += ['-v', f'{fixtures}:/cycle-fixtures',
                   '-v', f'{source_root / "b70_cycle_replay.py"}:/opt/b70-trace/b70_cycle_replay.py:ro',
                   '-v', f'{source_root / "b70_cycle_replay.pth"}:/opt/venv/lib/python3.12/site-packages/b70_cycle_replay.pth:ro',
                   '-e', 'B70_CYCLE_REPLAY=1', '-e', 'PYTHONPATH=/opt/b70-runtime:/opt/b70-trace']
    if capture:
        result += ['-v', f'{source_root / "b70_linear_capture.py"}:/opt/b70-trace/b70_linear_capture.py:ro',
                   '-v', f'{source_root / "b70_linear_capture.pth"}:/opt/venv/lib/python3.12/site-packages/b70_linear_capture.pth:ro',
                   '-e', 'B70_LINEAR_CAPTURE=1', '-e', 'PYTHONPATH=/opt/b70-runtime:/opt/b70-trace']
        arguments += ['--enforce-eager']
    if verified:
        if arm not in ('current', 'fixed', 'fused'):
            raise ValueError('verified startup requires an unmodified current or fixed image')
        return result + ['--entrypoint', 'python', image, '/opt/b70/verify_production.py', '--serve', *arguments]
    return result + ['--entrypoint', 'vllm', image, 'serve', *arguments]


def provenance(phase):
    # Maps establish mappings only; ELF dependencies are recorded separately.
    script = '''import glob,json,os,subprocess
workers=[]
for p in glob.glob('/proc/[0-9]*/cmdline'):
 try:
  cmd=open(p,'rb').read().replace(b'\\0',b' ').decode(errors='replace')
  if 'EngineCore' in cmd or 'multiprocessing.spawn' in cmd:
   maps=open(p.replace('cmdline','maps')).read()
   workers.append({'pid':p.split('/')[2],'cmdline':cmd,'library_maps':[l for l in maps.splitlines() if '.so' in l]})
 except OSError: pass
paths=glob.glob('/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/*xpu_C*.so')+glob.glob('/opt/b70/native_sdpa.so')
out={'workers':workers,'dependencies':{p:subprocess.run(['ldd',p],capture_output=True,text=True).stdout for p in paths}}
print(json.dumps(out))'''
    output = subprocess.check_output(['docker', 'exec', '-e', 'B70_DECODE_TRACE=0',
        '-e', 'B70_LINEAR_CAPTURE=0', '-e', 'B70_CYCLE_REPLAY=0', CONTAINER, 'python', '-c', script], text=True)
    save(phase / 'runtime-libraries.json', json.loads(output))


def trace_window(phase, case_dir, count, finished, errors):
    try:
        deadline = time.monotonic() + 120
        while True:
            ready = 0
            for index in range(count):
                path = case_dir / f'stream-{index + 1}.jsonl'
                if path.exists():
                    for line in path.read_text().splitlines():
                        try:
                            event = json.loads(line)['event']
                        except ValueError:
                            continue
                        if any((choice.get('delta') or {}).get('content') for choice in event.get('choices', [])):
                            ready += 1
                            break
            if ready == count:
                break
            if finished.is_set() or time.monotonic() > deadline:
                raise RuntimeError('no full-overlap trace window')
            time.sleep(.1)
        time.sleep(1)
        bench.http_json(BASE, '/start_profile', {})
        (phase / 'trace-active').touch()
        time.sleep(2)
        (phase / 'trace-active').unlink()
        bench.http_json(BASE, '/stop_profile', {}, timeout=180)
    except BaseException as error:
        errors.append(repr(error))
        (phase / 'trace-active').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--capture-linears', action='store_true')
    parser.add_argument('--probe-linears', action='store_true')
    parser.add_argument('--cycle-replay', action='store_true')
    parser.add_argument('--cycle-fixture-root', type=Path)
    parser.add_argument('--quality-task-sets', default='')
    parser.add_argument('--quality-arms', default='', help='Comma-separated quality arms; default: every selected arm')
    parser.add_argument('--mixed', action='store_true')
    parser.add_argument('--verified-startup', action='store_true')
    parser.add_argument('--arms', default='old,current,current,old,old,current')
    parser.add_argument('--scenarios', default='phase-4k-c1,concurrency-4k-c4')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.restore:
        restore(root)
        return
    root.mkdir(parents=True, exist_ok=False)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = LOCK.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    state = {'status': 'preflight', 'started_unix': time.time(), 'arms': []}
    save(root / 'state.json', state)
    item = inspect(PRODUCTION)
    if item['Image'] != IMAGES['current']:
        raise RuntimeError('unexpected production image')
    config = json.loads(bench.arg_after(item['Config']['Cmd'], '--speculative-config'))
    if config != {'method': 'mtp', 'num_speculative_tokens': 4}:
        raise RuntimeError('expected MTP4')
    metrics, _ = bench.metric_snapshot('http://127.0.0.1:8081')
    if metrics['running'] or metrics['waiting']:
        raise RuntimeError('production is busy')
    caps = list(Path('/sys/bus/pci/devices/0000:03:00.0/hwmon').glob('*/power1_cap'))
    if len(caps) != 1 or int(caps[0].read_text()) != 180000000:
        raise RuntimeError('power cap differs from 180 W')
    manifest = json.loads((REPO / 'config/frozen_fixture_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'frozen fixture mismatch: {name}')
    arms = args.arms.split(',')
    if any(arm not in IMAGES for arm in arms):
        raise ValueError('unknown arm')
    quality_arms = set(filter(None, args.quality_arms.split(','))) or set(arms)
    if not quality_arms.issubset(arms):
        raise ValueError('quality arms must be included in the run order')
    for image in set(IMAGES[arm] for arm in arms):
        actual = subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], text=True).strip()
        if actual != image:
            raise RuntimeError('image unavailable or identity mismatch')
    save(root / 'production-identity.json', {'image_id': item['Image'],
        'command': item['Config']['Cmd'], 'policy_sha256': item['Config']['Labels'].get('org.local.b70.policy.sha256')})
    save(root / 'provenance.json', {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'images': IMAGES, 'power_limit_w': 180, 'fixture_manifest_sha256': hashlib.sha256((REPO / 'config/frozen_fixture_manifest.json').read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'benchmark_sha256': hashlib.sha256((REPO / 'scripts/current-profile-benchmark.py').read_bytes()).hexdigest(),
        'profiled': args.profile, 'capture_linears_eager': args.capture_linears, 'cycle_replay': args.cycle_replay, 'cycle_fixture_root': str(args.cycle_fixture_root) if args.cycle_fixture_root else None, 'order': arms,
        'scenarios': args.scenarios.split(','), 'mixed': args.mixed,
        'quality_task_sets': args.quality_task_sets.split(',') if args.quality_task_sets else [],
        'quality_arms': sorted(quality_arms), 'verified_startup': args.verified_startup,
        'request_cache_policy': 'fresh worker per arm, distinct leading prompt markers for warmups and each scored wave; assert zero scored prefix hits',
        'host': subprocess.check_output(['uname', '-a'], text=True).strip()})
    sources, _ = bench.load_corpus(bench.CORPUS)
    selected = args.scenarios.split(',')
    scenarios = [row for row in bench.load_scenarios(REPO / 'benchmarks/current-profile-scenarios.json')
                 if row.name in selected]
    if len(scenarios) != len(set(selected)):
        raise ValueError('unknown or duplicate scenario')
    rows = []
    counts = {}
    try:
        subprocess.run(['systemctl', '--user', 'stop', SERVICE], check=True, timeout=100)
        for index, arm in enumerate(arms):
            repeat = counts.get(arm, 0)
            counts[arm] = repeat + 1
            if repeat >= 3:
                raise ValueError('only three frozen paired repetitions supported')
            phase = root / f'arm-{index + 1}-{arm}'
            phase.mkdir()
            cmd = command(item, arm, phase, root, args.profile, args.capture_linears, args.cycle_replay, args.verified_startup, args.cycle_fixture_root)
            save(phase / 'engine-command.json', cmd)
            state['status'] = f'arm-{index + 1}-{arm}-starting'
            save(root / 'state.json', state)
            print(state['status'], flush=True)
            with (phase / 'engine.log').open('w') as log:
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 1200
                    while not healthy(BASE):
                        if process.poll() is not None:
                            raise RuntimeError(f'{arm} startup exited {process.returncode}')
                        if time.monotonic() > deadline:
                            raise TimeoutError('diagnostic startup')
                        time.sleep(2)
                    for scenario in scenarios:
                        bench.run_once(BASE, CONTAINER, scenario, -1, phase / 'warmups',
                                       sources, bench.DEFAULT_PROMPT_NAMESPACE)
                    provenance(phase)
                    if args.cycle_replay:
                        (phase / 'cycle-replay-active').touch()
                    if args.capture_linears:
                        (phase / 'linear-capture-active').touch()
                    for scenario in scenarios:
                        bench.wait_idle(BASE)
                        finished, errors = threading.Event(), []
                        tracer = None
                        if args.profile:
                            tracer = threading.Thread(target=trace_window, args=(phase,
                                phase / f'{scenario.name}-r{repeat + 1}', scenario.concurrency,
                                finished, errors), daemon=True)
                            tracer.start()
                        row = bench.run_once(BASE, CONTAINER, scenario, repeat, phase,
                                             sources, bench.DEFAULT_PROMPT_NAMESPACE,
                                             fixture_root=FIXTURE)
                        if row['prompt_tokens_cached'] or row['preemptions'] or row['prefill_recompute_excess']:
                            raise RuntimeError('scored wave had cache hits, preemption or recomputation')
                        finished.set()
                        if tracer:
                            tracer.join(timeout=200)
                            if tracer.is_alive() or errors:
                                raise RuntimeError(f'trace failed: {errors}')
                        row.update(arm=arm, arm_index=index + 1,
                                   scored=not (args.profile or args.capture_linears or args.cycle_replay))
                        rows.append(row)
                        save(root / 'results.json', rows)
                        print(arm, scenario.name, 'round_ms', row['fully_overlapped_round_equivalent_ms'],
                              'acceptance', row['fully_overlapped_speculative_acceptance'], flush=True)
                    if args.mixed or args.quality_task_sets:
                        env = dict(os.environ, B70_FROZEN_FIXTURE_ROOT=str(FIXTURE))
                        experiment = REPO / 'benchmarks/experiments/onednn-prefill'
                        if args.mixed:
                            with (phase / 'mixed.log').open('w') as check_log:
                                subprocess.run([sys.executable, str(experiment / 'mixed_practical_128k.py'),
                                    '--base', BASE, '--container', CONTAINER, '--arm', arm,
                                    '--namespace', f'b70-decode-20260930-mixed-r{repeat + 1}',
                                    '--sampled-background', '--output', str(phase / 'mixed.json')],
                                    env=env, stdout=check_log, stderr=subprocess.STDOUT, check=True, timeout=1800)
                        for task_set in filter(None, args.quality_task_sets.split(',')) if arm in quality_arms else ():
                            with (phase / f'quality-{task_set}.log').open('w') as check_log:
                                subprocess.run([sys.executable, str(experiment / 'run_performance_tasks.py'),
                                    '--base', BASE, '--container', CONTAINER, '--arm', arm,
                                    '--task-set', task_set, '--output', str(phase / f'quality-{task_set}.jsonl')],
                                    env=env, stdout=check_log, stderr=subprocess.STDOUT, check=True, timeout=3600)
                    if args.cycle_replay:
                        for scenario in scenarios:
                            if not (phase / f'cycle-replay-c{scenario.concurrency}.json').exists():
                                raise RuntimeError('cycle replay did not produce a verified result')
                    if args.profile:
                        (phase / 'trace-flush').touch()
                        deadline = time.monotonic() + 30
                        while not list(phase.glob('trace-flushed-*')):
                            if time.monotonic() > deadline:
                                raise RuntimeError('worker step trace did not flush')
                            time.sleep(.2)
                finally:
                    subprocess.run(['docker', 'stop', '-t', '30', CONTAINER],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
                    process.wait(timeout=45)
            state['arms'].append({'index': index + 1, 'arm': arm, 'repeat': repeat + 1})
            save(root / 'state.json', state)
        if args.probe_linears:
            folders = list(root.glob('arm-*/linear-operands'))
            if len(folders) != 1:
                raise RuntimeError('linear probes require exactly one captured operand directory')
            probe_root = root / 'linear-probes'
            probe_root.mkdir()
            source_root = probe_root / 'sources'
            source_root.mkdir()
            for name in ('probe-decode-linears.py', 'probe-prefill-row-dispatch.py'):
                shutil.copyfile(REPO / 'scripts' / name, source_root / name)
            gpu_group = str(Path('/dev/dri/renderD128').stat().st_gid)
            for name, image, flags in (
                    ('old', IMAGES['old'], []), ('current-original', IMAGES['current'], []),
                    ('current-patched', IMAGES['current'], ['--patched']),
                    ('current-patched-sdpa', IMAGES['current'], ['--patched', '--load-sdpa']),
                    ('fixed', IMAGES['fixed'], ['--patched', '--load-sdpa']),
                    ('fused', IMAGES['fused'], ['--patched', '--load-sdpa'])):
                with (probe_root / f'{name}.log').open('w') as log:
                    subprocess.run(['docker', 'run', '--rm', '--network=none', '--device', '/dev/dri',
                        '--group-add', gpu_group, '-v', f'{folders[0]}:/operands:ro',
                        '-v', f'{source_root}:/scripts:ro', '-v', f'{probe_root}:/out',
                        '--entrypoint', 'python', image, '/scripts/probe-decode-linears.py',
                        '--operands', '/operands', '--output', f'/out/{name}.json', *flags],
                        stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
            with (probe_root / 'prefill-numeric.log').open('w') as log:
                subprocess.run(['docker', 'run', '--rm', '--network=none', '--device', '/dev/dri',
                    '--group-add', gpu_group, '-v', f'{folders[0]}:/operands:ro',
                    '-v', f'{source_root}:/scripts:ro', '-v', f'{probe_root}:/out',
                    '--entrypoint', 'python', IMAGES['fused'], '/scripts/probe-prefill-row-dispatch.py',
                    '--operands', '/operands', '--output', '/out/prefill-numeric.json'],
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['finished_unix'] = time.time()
        save(root / 'state.json', state)
        restore(root)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Reuse the frozen BF16 panel for separately identified target-runtime arms.

Raw full-vocabulary arrays stay in benchmark-results; commit small manifests
and comparisons separately. This scores prompt distributions, not throughput.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / 'benchmarks/experiments/quantization-reference'
PANEL_SHA = 'cbf1a71bbda470859f2c0786cb7134e260111d2072c4753a6732021dadae6171'
REFERENCE_SHA = '41dd8ce312ae11832360649657b936d8a937e8c33bfe6e1382eba083897ef884'
NAME = 'b70-exl3-quality-stages'
SERVICE = 'b70-qwen38-vllm.service'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(command, **kwargs):
    print(json.dumps({'command': command, 'unix': time.time()}), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def stage_config(stage):
    if stage == 'target-full-candidate':
        return dict(kv_cache_dtype='fp8', enforce_eager=False, enable_prefix_caching=True,
                    max_model_len=262144, max_num_seqs=16, max_num_batched_tokens=4096,
                    gpu_memory_utilization=.965, limit_mm_per_prompt={'image':32,'video':4},
                    mm_processor_kwargs={'max_pixels':4194304}, mamba_cache_mode='align',
                    scheduler_reserve_full_isl=True, watermark=0.0,
                    speculative_config={'method':'mtp','num_speculative_tokens':3},
                    compilation_config={'cudagraph_mode':'FULL_DECODE_ONLY',
                        'cudagraph_capture_sizes':[1,2,4,8,12,16,24,32,40,48,56,64]}), True
    engine = dict(kv_cache_dtype='auto', enforce_eager=True,
                  enable_prefix_caching=False, max_model_len=2048,
                  max_num_seqs=1, max_num_batched_tokens=1024,
                  gpu_memory_utilization=0.80,
                  limit_mm_per_prompt={'image': 0, 'video': 0})
    index = ('target-fp16', 'target-fp8', 'target-int8', 'target-graphs', 'target-mtp').index(stage)
    if index >= 1:
        engine['kv_cache_dtype'] = 'fp8'
    if index >= 3:
        engine['enforce_eager'] = False
        engine['compilation_config'] = {
            'cudagraph_mode': 'FULL_DECODE_ONLY',
            'cudagraph_capture_sizes': [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]}
    if index >= 4:
        engine['speculative_config'] = {'method': 'mtp', 'num_speculative_tokens': 3}
    return engine, index >= 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--reference', type=Path, default=REPO / 'benchmark-results/quantization-reference-20261001')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contract-campaign', type=Path, required=True,
                        help='Completed equal-contract qualification of this exact candidate image')
    parser.add_argument('--stages', nargs='+', default=['target-fp16', 'target-fp8', 'target-int8', 'target-graphs', 'target-mtp'])
    parser.add_argument('--windows', type=int, default=0, help='Nonzero is capture pilot only; no full-panel comparison')
    parser.add_argument('--reuse-completed', type=Path, help='Reuse completed matching arms from a terminal campaign; never repeat them')
    parser.add_argument('--expanded-contract-campaign', type=Path, help='Required for the full expanded serving precision arm')
    args = parser.parse_args()
    configs = {stage: stage_config(stage) for stage in args.stages}
    if len(configs) != len(args.stages):
        raise ValueError('Duplicate stages')
    reference = args.reference.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError('Fresh campaign directory required')
    if sha(reference / 'panel.json') != PANEL_SHA or sha(reference / 'reference-bf16.npz') != REFERENCE_SHA:
        raise RuntimeError('Frozen panel/BF16 bundle identity mismatch')
    image = subprocess.check_output(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    production = json.loads((REPO / 'config/production_image.json').read_text())
    if image == production['image_id']:
        raise RuntimeError('A distinct EXL3 candidate image is required')
    checkpoint = Path.home() / '.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw'
    draft_vocab = '/opt/exl3xpu/models/qwen3.8-27b-exl3-4.00bpw/draft_vocab.json'
    if not (checkpoint / 'config.json').is_file():
        raise RuntimeError('Pinned local checkpoint is missing')
    lockpath = REPO.parent / 'Local-AI-B70/qwen38/context-benchmark/run.lock'
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open('a') as lock:
        # Block behind a live qualification worker; never overlap GPU campaigns.
        fcntl.flock(lock, fcntl.LOCK_EX)
        contract_file = args.contract_campaign.resolve() / 'campaign.json'
        contract = json.loads(contract_file.read_text())
        required_cases = {'regression', 'smoke', '103k', '139k', '188k', 'near-limit',
                          'c4-long', 'image-long', 'extension-abort-long'}
        if contract['status'] != 'COMPLETE' or contract['image'] != image or not required_cases.issubset(contract['cases']):
            raise RuntimeError('Full equal-contract qualification of this image must complete first')
        for case in required_cases:
            result = contract['cases'][case]['result']
            if result['status'] not in {'COMPLETE', 'PASS'}:
                raise RuntimeError('Incomplete contract case: ' + case)
            if case in {'103k', '139k', '188k', 'near-limit', 'c4-long', 'image-long'} and result['native']['preemptions'] != 0:
                raise RuntimeError('Unresolved preemption in contract case: ' + case)
        if 'target-full-candidate' in configs:
            if not args.expanded_contract_campaign:
                raise RuntimeError('Expanded candidate requires its own completed capacity qualification')
            expanded = json.loads((args.expanded_contract_campaign / 'campaign.json').read_text())
            needed = {'regression','smoke','near-limit','c16-long','image-long','extension-abort-long'}
            wanted = dict(max_model_len=262144,max_num_seqs=16,max_num_batched_tokens=4096,
                          gpu_memory_utilization=.965,limit_mm_per_prompt={'image':32,'video':4})
            if (expanded['status'] != 'COMPLETE' or expanded['image'] != image
                    or not needed.issubset(expanded['cases'])
                    or any(expanded['settings_overrides'].get(k) != v for k,v in wanted.items())):
                raise RuntimeError('Expanded capacity qualification is incomplete or mismatched')
            for case in needed:
                result = expanded['cases'][case]['result']
                if result['status'] not in {'COMPLETE', 'PASS'}:
                    raise RuntimeError('Incomplete expanded capacity case: '+case)
                if case in ('near-limit','c16-long','image-long') and result['native']['preemptions'] != 0:
                    pressure_allowed = (case == 'c16-long' and expanded.get('allow_c16_preemptions')
                                        and result.get('preemption_policy') == 'record-pressure')
                    if not pressure_allowed:
                        raise RuntimeError('Unresolved expanded-profile preemption: '+case)
        output.mkdir(parents=True)
        (output / 'panel.json').write_bytes((reference / 'panel.json').read_bytes())
        (output / 'configs').mkdir()
        for label in ('bf16', 'fp16', 'gptq', 'exl3'):
            (output / label).symlink_to(reference / label, target_is_directory=True)
        state = {'schema': 1, 'status': 'RUNNING', 'image_id': image,
                 'contract_campaign_sha256': sha(contract_file),
                 'expanded_contract_campaign_sha256': sha(args.expanded_contract_campaign/'campaign.json') if args.expanded_contract_campaign else None,
                 'reference_panel_sha256': PANEL_SHA, 'reference_bundle_sha256': REFERENCE_SHA,
                 'reference_file_sha256': {str(p.relative_to(reference)): sha(p)
                     for p in sorted((reference / 'bf16').glob('window-*.npy'))},
                 'source_sha256': {str(p.relative_to(REPO)): sha(p)
                     for p in [Path(__file__), *sorted(SCRIPTS.glob('*.py'))]},
                 'model_config_sha256': sha(checkpoint / 'config.json'),
                 'tokenizer_sha256': sha(checkpoint / 'tokenizer.json'),
                 'stages': {}, 'started_unix': time.time(),
                 'scope': 'Staged short-panel prompt quality; long suffixes and generated decode/graph checks remain separate gates.',
                 'production_restored': False}

        def save():
            (output / 'campaign.json').write_text(json.dumps(state, indent=2) + '\n')

        def interrupted(signum, frame):
            raise InterruptedError(f'Signal {signum}')

        signal.signal(signal.SIGINT, interrupted)
        signal.signal(signal.SIGTERM, interrupted)
        save()
        try:
            run(['systemctl', '--user', 'stop', SERVICE], timeout=90)
            run(['docker', 'run', '--rm', '--network', 'none', '--memory', '4g',
                 '-v', str(SCRIPTS) + ':/scripts:ro', '-v', str(reference) + ':/reference:ro',
                 '-v', str(output) + ':/results', '--entrypoint', 'python', image,
                 '/scripts/validate_reference_bundle.py', '--root', '/reference',
                 '--output', '/results/reference-integrity.json'])
            state['reference_integrity_sha256'] = sha(output / 'reference-integrity.json')
            state['draft_vocab_sha256'] = subprocess.check_output(
                ['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'sha256sum', image, draft_vocab],
                text=True).split()[0]
            previous = None
            reuse = args.reuse_completed.resolve() if args.reuse_completed else None
            if reuse:
                previous = json.loads((reuse / 'campaign.json').read_text())
                if previous['status'] not in {'FAILED', 'COMPLETE'} or previous['image_id'] != image:
                    raise RuntimeError('Only terminal campaigns of the same image may supply completed arms')
                state['reuse_campaign_sha256'] = sha(reuse / 'campaign.json')
            for stage, (config, int8) in configs.items():
                old = previous['stages'].get(stage) if previous else None
                if old and old['status'] in {'COMPLETE', 'REUSED'}:
                    summary_file = reuse / stage / 'summary.json'
                    summary = json.loads(summary_file.read_text())
                    if (old['engine_config'] != config or old['int8_prefill'] != int8
                            or summary['panel_sha256'] != PANEL_SHA or len(summary['windows']) != 16
                            or sha(summary_file) != old['summary_sha256']):
                        raise RuntimeError('Completed-arm provenance mismatch: ' + stage)
                    actual_source = (reuse / stage).resolve()
                    (output / stage).symlink_to(actual_source, target_is_directory=True)
                    state['stages'][stage] = {'status': 'REUSED', 'source': str(actual_source),
                                             'engine_config': config, 'int8_prefill': int8,
                                             'summary_sha256': sha(summary_file)}
                    save()
                    continue
                config_path = output / 'configs' / (stage + '.json')
                config_path.write_text(json.dumps(config, indent=2) + '\n')
                cache = Path.home() / '.cache/exl3xpu/migration-quality' / image.removeprefix('sha256:') / stage
                for directory in ('vllm', 'triton', 'neo_compiler_cache'):
                    (cache / directory).mkdir(parents=True, exist_ok=True)
                env = {'HF_HUB_OFFLINE': '1', 'PYTHONPATH': '/scripts',
                       'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
                       'ZE_FLAT_DEVICE_HIERARCHY': 'COMPOSITE', 'ZE_AFFINITY_MASK': '0',
                       'OMP_NUM_THREADS': '4', 'EXL3_TARGET_RUNTIME': 'vllm030',
                       'EXL3_ONEDNN_ATTN': '0', 'EXL3_INT8_PREFILL': str(int(int8)),
                       'EXL3_DRAFT_VOCAB': draft_vocab,
                       'VLLM_XPU_ENABLE_XPU_GRAPH': str(int(not config['enforce_eager'])),
                       'EXL3_LOADER_REPORT_DIR': '/results/loader-' + stage,
                       'B70_ONEDNN_PREFILL': '0', 'B70_ONEDNN_MIXED_ROUTE': '0',
                       'B70_GPTQ_W4A8_PREFILL': '0'}
                command = ['docker', 'run', '--rm', '--name', NAME,
                           '--device', '/dev/dri', '--network', 'bridge',
                           '--memory', '12g', '--shm-size', '4g',
                           '-v', '/dev/dri/by-path:/dev/dri/by-path:ro',
                           '-v', str(checkpoint) + ':/exl3:ro',
                           '-v', str(SCRIPTS) + ':/scripts:ro',
                           '-v', str(output) + ':/results',
                           '-v', str(cache / 'vllm') + ':/root/.cache/vllm',
                           '-v', str(cache / 'triton') + ':/root/.triton/cache',
                           '-v', str(cache / 'neo_compiler_cache') + ':/root/.cache/neo_compiler_cache']
                for key, value in env.items():
                    command += ['-e', key + '=' + value]
                command += ['--entrypoint', 'python', image, '-u', '/scripts/run_native.py',
                            '--model', '/exl3', '--quantization', 'exl3',
                            '--panel', '/results/panel.json', '--out', '/results/' + stage,
                            '--engine-config', '/results/configs/' + stage + '.json']
                if args.windows:
                    command += ['--windows', str(args.windows)]
                state['stages'][stage] = {'status': 'RUNNING', 'command': command,
                                         'engine_config': config, 'int8_prefill': int8}
                save()
                with (output / (stage + '.log')).open('w') as log:
                    run(command, stdout=log, stderr=subprocess.STDOUT)
                state['stages'][stage]['status'] = 'COMPLETE'
                state['stages'][stage]['summary_sha256'] = sha(output / stage / 'summary.json')
                save()
            if not args.windows:
                with (output / 'comparison.log').open('w') as log:
                    comparison_command = ['docker', 'run', '--rm', '--network', 'none', '--memory', '4g',
                         '-v', str(SCRIPTS) + ':/scripts:ro',
                         '-v', str(output) + ':' + str(output),
                         '-v', str(reference) + ':' + str(reference) + ':ro',
                         '-v', str(checkpoint / 'tokenizer.json') + ':/tokenizer.json:ro']
                    reused_roots = {Path(s['source']).parent for s in state['stages'].values() if s['status']=='REUSED'}
                    for reused_root in sorted(reused_roots):
                        comparison_command += ['-v', str(reused_root) + ':' + str(reused_root) + ':ro']
                    comparison_command += [
                         '--entrypoint', 'python', image, '/scripts/compare.py', '--root', str(output),
                         '--arms', 'fp16', 'gptq', 'exl3', *args.stages,
                         '--tokenizer', '/tokenizer.json',
                         '--output-name', 'comparison.json']
                    run(comparison_command, stdout=log, stderr=subprocess.STDOUT)
            state['status'] = 'COMPLETE'
        except BaseException as exc:
            state['status'] = 'FAILED'
            state['error'] = repr(exc)
            for stage in state['stages'].values():
                if stage['status'] == 'RUNNING':
                    stage['status'] = 'FAILED'
            save()
            raise
        finally:
            subprocess.run(['docker', 'rm', '-f', NAME], capture_output=True, timeout=30)
            run(['systemctl', '--user', 'stop', SERVICE], timeout=90)
            state['production_left_offline'] = True
            state['finished_unix'] = time.time()
            save()
            print(json.dumps({'status': state['status'], 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()

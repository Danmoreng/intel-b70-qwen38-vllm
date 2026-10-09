#!/usr/bin/env python3
"""Bounded actual-worker diagnostics with exact production restoration."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO/'scripts'))
from exl3_candidate_worker import Worker, BASE, sha

IMAGE = 'sha256:b0b9244420a2586f8328b14aeff21d94673632ad7342032a285f3b877f2de49b'
PRODUCTION = json.loads((REPO/'config/experiments/vllm-031-upgrade/production_image.json').read_text())['image_id']
DRAFT = Path.home()/'.cache/exl3xpu/incoai-Qwen3.8-27B-DFlash2-dedf8df6-fp16'
FIXTURE = REPO.parent/'intel-b70-qwen38-vllm/benchmark-results/meaningful-full-profile/run-20260923-201101-w0.00'


def api(path, data=None, timeout=900):
    request = urllib.request.Request(BASE+path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--ledger-only', action='store_true')
    args = parser.parse_args()
    root = REPO/'benchmark-results/dflash2-pro-diagnostics-v031'/args.name
    root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((REPO/'config/experiments/exl3-migration/full-serving-fixture-manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(FIXTURE/name) == digest, name
    conversion = json.loads((DRAFT/'conversion.json').read_text())
    assert sha(DRAFT/'model.safetensors') == conversion['fp16_sha256']
    assert sha(DRAFT/'config.json') == conversion['config_sha256']
    state = dict(status='RUNNING', image=IMAGE, production_image=PRODUCTION,
        started_unix=time.time(), parameters=dict(B=8, V=7, max_model_len=96000,
        target_dtype='float16', draft_dtype='float16', kv_dtype='fp8', power_w=180),
        diagnostic_only=True, baseline_regenerated=False,
        sources={p.name: sha(p) for p in HERE.glob('*') if p.is_file()})
    for p in HERE.glob('*'):
        if p.is_file():
            (root/p.name).write_bytes(p.read_bytes())
    snapshot = root/'source-snapshot'; snapshot.mkdir()
    for source in (HERE, HERE.parent/'dflash2-serving-v031'):
        for p in source.glob('*.py'):
            target = snapshot/p.name
            assert not target.exists(), p.name
            target.write_bytes(p.read_bytes())
    def save():
        (root/'campaign.json').write_text(json.dumps(state, indent=2)+'\n')
    restore_spec = importlib.util.spec_from_file_location('restore', REPO/'scripts/run-exl3-review-micro.py')
    restore = importlib.util.module_from_spec(restore_spec); restore_spec.loader.exec_module(restore)
    worker = Worker(IMAGE, root/'worker', name='b70-dflash2-pro-diagnostics',
        env={'EXL3_SDPA_CACHE_CAPACITY': '64', 'VLLM_SERVER_DEV_MODE': '1',
             'PYTHONPATH': '/experiment:/serving-inspect:/opt/b70-runtime'},
        binds=[(snapshot, '/experiment'), (DRAFT, '/draft')])
    argv = worker.profile['argv']
    argv[argv.index('--max-model-len')+1] = '96000'
    argv[argv.index('--speculative-config')+1] = json.dumps(dict(method='dflash', model='/draft', num_speculative_tokens=7))
    argv += ['--worker-extension-cls', 'diagnostics.WorkerExtension']
    def rpc(method, **kwargs):
        result = api('/collective_rpc', dict(method=method, args=[], kwargs=kwargs))
        (root/f'{method}.json').write_text(json.dumps(result, indent=2)+'\n')
        print('RPC_COMPLETE', method, flush=True)
        return result
    def wave(name):
        prompt = (FIXTURE/'phase-32k-c1-r1/prompt-1.txt').read_text()
        body = dict(model='Qwen3.8-27B', messages=[dict(role='user', content=prompt)],
            temperature=1.0, top_p=0.95, top_k=20, seed=770000, max_tokens=384, ignore_eos=True,
            chat_template_kwargs={'enable_thinking': False})
        (root/f'{name}-request.json').write_text(json.dumps(body, ensure_ascii=False)+'\n')
        api('/reset_prefix_cache', {})
        print('START_DIAGNOSTIC_WAVE', name, flush=True)
        response = api('/v1/chat/completions', body)
        assert response['usage']['completion_tokens'] == 384, response['usage']
        (root/f'{name}-response.json').write_text(json.dumps(response, indent=2)+'\n')
        print('COMPLETE_DIAGNOSTIC_WAVE', name, response['usage'], flush=True)
    with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        save()
        try:
            subprocess.run(['systemctl', '--user', 'stop', 'b70-qwen38-vllm.service'], check=True, timeout=90)
            assert subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).splitlines() == ['b70-chatui']
            print('START_WORKER', flush=True); worker.start()
            rpc('inspect_dflash_setup'); rpc('cache_ledger')
            if not args.ledger_only:
                rpc('install_event_profile')
                rpc('begin_event_profile'); wave('events')
                rpc('finish_event_profile', path='/results/event-profile.json')
                rpc('begin_kernel_trace', cycles=8); wave('kernel-trace')
                rpc('finish_kernel_trace', path='/results/kernel-trace.json')
                rpc('install_round_capture'); rpc('begin_round_capture'); wave('tensor-capture')
                rpc('finish_round_capture')
                rpc('rejection_statistical_test')
            state['status'] = 'DIAGNOSTIC_COLLECTION_COMPLETE_NOT_QUALIFICATION'
        except BaseException as error:
            state.update(status='FAILED', error=repr(error)); raise
        finally:
            worker.stop()
            subprocess.run(['systemctl', '--user', 'start', 'b70-qwen38-vllm.service'], check=True, timeout=90)
            state['restoration'] = restore.wait_for_qualified_service(PRODUCTION, timeout=900)
            state['finished_unix'] = time.time(); save()
            print('RESTORED_QUALIFIED_PRODUCTION', flush=True)


if __name__ == '__main__':
    main()

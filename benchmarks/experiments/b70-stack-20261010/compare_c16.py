#!/usr/bin/env python3
"""Named missing reference: matched C16 queue pressure, not old requalification."""
import fcntl
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
from exl3_candidate_worker import BASE, Worker, sha


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    root = REPO / 'benchmark-results/b70-stack-20261010/E01-c16-matched'
    root.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((HERE / 'baseline-manifest.json').read_text())['production']['image_id']
    candidate = json.loads((REPO / 'config/experiments/b70-stack-20261010/production_image.json').read_text())['image_id']
    D = module('c16_contract', 'run-exl3-contract-diagnostics.py')
    restore = module('c16_restore', 'run-exl3-review-micro.py')
    library = json.loads((REPO / 'benchmark-results/b70-stack-20261010/E01-oneDNN-build/library-receipt.json').read_text())
    snapshot = root / 'diagnostic-sources'
    snapshot.mkdir()
    for name in ['inventory.py', 'profile_capture.py']:
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    state = dict(status='RUNNING', baseline=baseline, candidate=candidate, blocks=[],
                 started_unix=time.time(), source_sha256=sha(Path(__file__)),
                 scope='Named missing contemporaneous C16-pressure reference after candidate recorded 3 versus archived 2 preemptions, without recompute. Three balanced paired fresh workers, identical API/media/C4 antecedent cases and same 16x8192/256 queue workload. Not a historical baseline correctness or full serving campaign.')

    def save():
        (root / 'campaign.json').write_text(json.dumps(state, indent=2) + '\n')

    worker = None
    with (REPO.parent / 'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            subprocess.run(['systemctl', '--user', 'stop', 'b70-qwen38-vllm.service'], check=True, timeout=90)
            assert subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).splitlines() == ['b70-chatui']
            save()
            for index in range(3):
                block = dict(index=index, order=['baseline', 'candidate'] if index % 2 == 0 else ['candidate', 'baseline'], arms={})
                state['blocks'].append(block)
                save()
                for role in block['order']:
                    destination = root / f'block-{index}-{role}'
                    worker = Worker(baseline if role == 'baseline' else candidate, destination,
                        'b70-stack-c16', env={'EXL3_MIGRATION_TRACE': '/results/trace', 'VLLM_SERVER_DEV_MODE': '1',
                                            'PYTHONPATH': '/experiment:/opt/b70-runtime'}, binds=[(snapshot, '/experiment')])
                    worker.profile['argv'] += ['--worker-extension-cls', 'inventory.WorkerExtension']
                    worker.start()
                    kwargs = dict(label='initial')
                    if role == 'candidate':
                        kwargs.update(expected_onednn_sha256=library['library_sha256'], expected_onednn_version=library['version'])
                    D.R.http(BASE, '/collective_rpc', dict(method='inventory', args=[], kwargs=kwargs))
                    inventory = json.loads((worker.root / 'inventory-initial.json').read_text())
                    assert inventory['num_blocks'] == 200 and inventory['kv_pool_bytes'] == 11141120000
                    arm = dict(cases={}, inventory_sha256=sha(worker.root / 'inventory-initial.json'),
                               num_blocks=inventory['num_blocks'], kv_pool_bytes=inventory['kv_pool_bytes'])
                    block['arms'][role] = arm
                    save()
                    cases = [('api', lambda path: D.smoke(path, 262144, {'image': 32, 'video': 4})),
                             ('media', lambda path: D.media_case(path, {'image': 32, 'video': 4})),
                             ('c4-long', lambda path: D.operational('c4-long', path, True)),
                             ('c16-long', lambda path: D.operational('c16-long', path, True))]
                    for name, operation in cases:
                        target = destination / name
                        target.mkdir()
                        arm['cases'][name] = operation(target)
                        save()
                        if name == 'c16-long':
                            native = arm['cases'][name]['native']
                            print('C16', index, role, 'preemptions', native['preemptions'],
                                  'computed', native['prefill_tokens'], 'logical', native['prompt_tokens'], flush=True)
                    worker.stop()
                    worker = None
                assert block['arms']['baseline']['cases']['c16-long']['request_sha256'] == block['arms']['candidate']['cases']['c16-long']['request_sha256']
            state['status'] = 'COMPLETE_MATCHED_C16_PRESSURE_REVIEW_REQUIRED'
        except BaseException as error:
            state.update(status='FAILED', error=repr(error))
            raise
        finally:
            if worker is not None:
                worker.stop()
            subprocess.run(['systemctl', '--user', 'start', 'b70-qwen38-vllm.service'], check=True, timeout=90)
            state['restoration'] = restore.wait_for_qualified_service(baseline, timeout=900)
            state['finished_unix'] = time.time()
            save()
            print('RESTORED_QUALIFIED_PRODUCTION', flush=True)


if __name__ == '__main__':
    main()

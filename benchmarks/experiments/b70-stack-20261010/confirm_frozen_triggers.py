#!/usr/bin/env python3
"""Measure only named missing contemporary references for final review triggers.

This cannot launch before the complete final matrix/export. It is not a
historical baseline requalification or a second depth/dependency matrix.
"""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raw = REPO / 'benchmark-results/b70-stack-20261010'
    comparison_path = REPO / 'benchmarks/results/b70-stack-20261010/benchmark-comparison.json'
    comparison = json.loads(comparison_path.read_text())
    assert json.loads((raw / 'E01-final-controller-r3.json').read_text())['status'] == 'COMPLETE_FINAL_OPERATIONS_AND_FROZEN_SERVING_REVIEW_REQUIRED'
    assert json.loads((raw / 'E01-serving-final/campaign.json').read_text())['status'] == 'COMPLETE_REVIEW_SERVING_MATRIX'
    allowed = {row['name'] for row in comparison['scenarios'] if row['review_trigger']}
    selected = set(args.case)
    assert selected and selected <= allowed and len(selected) == len(args.case)
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    baseline, candidate = comparison['baseline_image_id'], comparison['candidate_image_id']
    assert baseline == json.loads((REPO / 'config/production_image.json').read_text())['image_id']
    D = module('frozen_contract', 'run-exl3-contract-diagnostics.py')
    restore = module('frozen_restore', 'run-exl3-review-micro.py')
    library = json.loads((raw / 'E01-oneDNN-build/library-receipt.json').read_text())
    snapshot = root / 'diagnostic-sources'
    snapshot.mkdir()
    for name in ['inventory.py', 'profile_capture.py']:
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    state = dict(status='RUNNING', baseline=baseline, candidate=candidate,
                 cases=args.case, independent_blocks=3, blocks=[], started_unix=time.time(),
                 source_sha256=sha(Path(__file__)), comparison_sha256=sha(comparison_path),
                 benchmark_sha256=sha(REPO / 'scripts/current-profile-benchmark.py'),
                 scope='Three balanced fresh-worker pairs for explicitly named historical final-matrix review triggers only. Original frozen prompt bytes, all original repeats, seeds, sampling, 1024 output budget and warmups retained. No profiling hooks installed; read-only inventory is outside measurement. Not old full-matrix or quality requalification.')

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
                        'b70-stack-frozen-reference', env={'EXL3_SDPA_CACHE_CAPACITY': '64',
                        'VLLM_SERVER_DEV_MODE': '1', 'PYTHONPATH': '/experiment:/opt/b70-runtime'},
                        binds=[(snapshot, '/experiment')])
                    worker.profile['argv'] += ['--worker-extension-cls', 'inventory.WorkerExtension']
                    worker.start()
                    kwargs = dict(label='initial')
                    if role == 'candidate':
                        kwargs.update(expected_onednn_sha256=library['library_sha256'], expected_onednn_version=library['version'])
                    D.R.http(BASE, '/collective_rpc', dict(method='inventory', args=[], kwargs=kwargs))
                    inventory = json.loads((worker.root / 'inventory-initial.json').read_text())
                    assert inventory['num_blocks'] == 200 and inventory['kv_pool_bytes'] == 11141120000
                    assert all('bfloat16' not in dtype for role_dtypes in inventory['registered_tensor_dtypes'].values() for dtype in role_dtypes)
                    arm = dict(status='RUNNING', inventory_sha256=sha(worker.root / 'inventory-initial.json'),
                               num_blocks=200, kv_pool_bytes=11141120000)
                    block['arms'][role] = arm
                    save()
                    command = [sys.executable, str(REPO / 'scripts/current-profile-benchmark.py'),
                               '--base', BASE, '--container', worker.name, '--expected-max-num-seqs', '16',
                               '--fixture-root', str(REPO.parent / 'intel-b70-qwen38-vllm/benchmark-results/meaningful-full-profile/run-20260923-201101-w0.00'),
                               '--legacy-prefix-namespace', '--output-root', str(destination / 'measured'), '--execute']
                    for case in args.case:
                        command += ['--only', case]
                    arm['command'] = command
                    save()
                    with (destination / 'benchmark.log').open('w') as log:
                        subprocess.run(command, check=True, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                    results = list((destination / 'measured').glob('*/results.json'))
                    assert len(results) == 1
                    measured = json.loads(results[0].read_text())
                    assert {case['scenario']['name'] for case in measured['cases']} == selected
                    assert all(case['preemptions'] == 0 and case['prefill_recompute_excess'] == 0 for case in measured['cases'])
                    arm.update(status='COMPLETE', results_sha256=sha(results[0]),
                               results_path=str(results[0].relative_to(root)))
                    save()
                    worker.stop()
                    worker = None
                    print('COMPLETE', index, role, args.case, flush=True)
            state['status'] = 'COMPLETE_NAMED_FROZEN_REFERENCES_REVIEW_REQUIRED'
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


if __name__ == '__main__':
    main()

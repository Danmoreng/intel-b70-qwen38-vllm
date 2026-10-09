#!/usr/bin/env python3
"""Repeat reviewed new-image load points; never run the stored old baseline."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts'))
from exl3_candidate_worker import Worker, BASE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decision', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    decision = json.loads(a.decision.read_text())
    assert decision['status'] == 'REQUIRES_PERFORMANCE_REVIEW'
    cfg = REPO / 'config/experiments/vllm-031-upgrade'
    release = json.loads((cfg / 'production_image.json').read_text())
    assert decision['image_id'] == release['image_id']
    selected = decision['performance_review_triggers']
    assert selected and not a.out.exists()
    lock = REPO.parent / 'Local-AI-B70/qwen38/context-benchmark/run.lock'
    with lock.open('a+') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.run(['systemctl', '--user', 'is-active', 'b70-qwen38-vllm.service'], capture_output=True, text=True)
        assert active.stdout.strip() != 'active'
        a.out.mkdir(parents=True)
        state = dict(status='RUNNING', image_id=release['image_id'],
                     baseline_regenerated=False, selected=selected,
                     decision_sha256=hashlib.sha256(a.decision.read_bytes()).hexdigest(),
                     started_unix=time.time())
        worker = Worker(release['image_id'], a.out / 'worker', name='b70-exl3-performance-review')
        command = [sys.executable, str(REPO / 'scripts/current-profile-benchmark.py'),
                   '--base', BASE, '--container', worker.name, '--expected-max-num-seqs', '16',
                   '--fixture-root', str(REPO.parent / 'intel-b70-qwen38-vllm/benchmark-results/meaningful-full-profile/run-20260923-201101-w0.00'),
                   '--legacy-prefix-namespace', '--output-root', str(a.out / 'measurements'), '--execute']
        for name in selected:
            command.extend(['--only', name])
        state['argv'] = command
        def save():
            (a.out / 'campaign.json').write_text(json.dumps(state, indent=2) + '\n')
        save()
        try:
            worker.start()
            with (a.out / 'benchmark.log').open('w') as log:
                subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
            raw_path, = (a.out / 'measurements').glob('run-*/results.json')
            raw = json.loads(raw_path.read_text())
            assert json.loads((a.out / 'worker/identity.json').read_text())['Image'] == release['image_id']
            assert raw['manifest']['container']['image'] == release['image_id']
            assert set(x['scenario']['name'] for x in raw['cases']) == set(selected)
            assert all(x['preemptions'] == 0 and x['all_prompt_counts_match'] and
                       x['all_completion_counts_exact'] and x['all_finish_reasons_length'] and
                       x['prefill_recompute_excess'] == 0 for x in raw['cases'])
            frozen = json.loads((REPO / 'config/experiments/exl3-migration/full-serving-fixture-manifest.json').read_text())['files']
            for case in raw['cases']:
                folder = raw_path.parent / f"{case['scenario']['name']}-r{case['repeat']}"
                for index in range(1, case['scenario']['concurrency'] + 1):
                    for filename in (f'prompt-{index}.txt', f'request-{index}.json'):
                        path = folder / filename
                        relative = str(path.relative_to(raw_path.parent))
                        if hashlib.sha256(path.read_bytes()).hexdigest() != frozen[relative]:
                            original = Path(raw['manifest']['fixture_root']) / relative
                            assert filename.endswith('.json')
                            assert hashlib.sha256(original.read_bytes()).hexdigest() == frozen[relative]
                            assert json.loads(path.read_text()) == json.loads(original.read_text())
            state.update(status='COMPLETE_NEW_IMAGE_PERFORMANCE_REPEATS',
                         raw_results_path=str(raw_path.resolve().relative_to(REPO)),
                         raw_results_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
        except BaseException as e:
            state.update(status='FAILED', error=repr(e))
            raise
        finally:
            worker.stop()
            state['finished_unix'] = time.time()
            save()


if __name__ == '__main__':
    main()

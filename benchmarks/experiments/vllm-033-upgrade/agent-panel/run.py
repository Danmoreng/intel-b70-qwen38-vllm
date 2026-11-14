#!/usr/bin/env python3
"""Run the frozen four-task agent panel against an already inspected worker."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'container', 'image'):
        p.add_argument('--' + name, required=True)
    for name in ('out', 'launch', 'runtime-source', 'worker-source'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    item = json.loads(subprocess.check_output(['docker', 'inspect', a.container], text=True))[0]
    assert item['Image'] == a.image
    inspect_source = """import hashlib,json,pathlib
paths={'native_binary_sha256':'/opt/exl3xpu/exl3xpu/_C.so',
       'cow_fix_sha256':'/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/b70_cow_copy_fix.py'}
print(json.dumps({k:hashlib.file_digest(pathlib.Path(p).open('rb'),'sha256').hexdigest() for k,p in paths.items()}))
"""
    receipt = json.loads(subprocess.check_output(
        ['docker', 'exec', a.container, 'python', '-c', inspect_source], text=True))
    receipt.update(image_id=item['Image'], container_id=item['Id'])
    identity = a.out / 'identity.json'
    identity.write_text(json.dumps(receipt, indent=2) + '\n')
    here = Path(__file__).resolve().parent
    report = {'status': 'RUNNING', 'image_id': a.image, 'tasks': []}

    def save():
        (a.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    try:
        for family in ('rangeset', 'dependencyplan'):
            fixture = here / 'fixtures' / family
            manifest = json.loads((fixture / 'manifest.json').read_text())
            for seed in manifest['seeds']:
                out = a.out / (family + '-' + str(seed))
                argv = [sys.executable, '-u', str(here / 'run_task.py'), '--fixture', str(fixture),
                        '--out', str(out), '--seed', str(seed), '--wall-seconds', '900',
                        '--base', a.base, '--container', a.container, '--image', a.image,
                        '--identity-receipt', str(identity), '--launch-receipt', str(a.launch),
                        '--runtime-source', str(a.runtime_source), '--worker-source', str(a.worker_source)]
                print('START_AGENT', family, seed, flush=True)
                with out.with_suffix('.log').open('w') as log:
                    subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=960)
                result = json.loads((out / 'summary.json').read_text())
                report['tasks'].append(result)
                save()
                print('END_AGENT', family, seed, result['status'], flush=True)
        assert len(report['tasks']) == 4
        report['successful_tasks'] = sum(bool(task['full_success']) for task in report['tasks'])
        assert report['successful_tasks'] == 4, 'Independent hidden acceptance failed'
        report['status'] = 'PASS_FROZEN_AGENT_PANEL_4_OF_4'
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()

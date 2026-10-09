#!/usr/bin/env python3
"""Small isolated sampler test; no target weights or old baseline generation."""
import fcntl
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
IMAGE = 'sha256:b0b9244420a2586f8328b14aeff21d94673632ad7342032a285f3b877f2de49b'
PRODUCTION = json.loads((REPO/'config/experiments/vllm-031-upgrade/production_image.json').read_text())['image_id']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['statistical', 'graph-attention'], default='statistical')
    args = parser.parse_args()
    root = REPO/'benchmark-results/dflash2-pro-diagnostics-v031'/time.strftime('sampler-%Y%m%d-%H%M%S')
    root.mkdir(parents=True)
    snapshot = root/'source-snapshot'; snapshot.mkdir()
    for p in HERE.glob('*.py'):
        (snapshot/p.name).write_bytes(p.read_bytes())
    receipt = dict(status='RUNNING', image=IMAGE, production_image=PRODUCTION)
    spec = importlib.util.spec_from_file_location('restore', REPO/'scripts/run-exl3-review-micro.py')
    restore = importlib.util.module_from_spec(spec); spec.loader.exec_module(restore)
    code = "import torch,json; torch.set_num_threads(2); from rejection_reference import gpu_test; from pathlib import Path; result=gpu_test(); Path('/results/rejection-statistics.json').write_text(json.dumps(result,indent=2)+'\\n'); print(result['status'],flush=True)"
    if args.suite == 'graph-attention':
        code = "import torch,json; torch.set_num_threads(2); from pathlib import Path; from rejection_reference import graph_test; from draft_attention_reference import gpu_test; a=graph_test(); Path('/results/proposal-graph.json').write_text(json.dumps(a,indent=2)+'\\n'); print(a['status'],flush=True); b=gpu_test(); Path('/results/draft-attention.json').write_text(json.dumps(b,indent=2)+'\\n'); print(b['status'],flush=True)"
    command = ['docker', 'run', '--rm', '--name', 'b70-dflash-sampler-probe',
        '--device', '/dev/dri', '-e', 'ZE_AFFINITY_MASK=0', '-e', 'PYTHONPATH=/experiment:/opt/exl3xpu',
        '-v', str(snapshot)+':/experiment:ro', '-v', str(root)+':/results',
        '--entrypoint', '/opt/venv/bin/python', IMAGE, '-c', code]
    receipt['command'] = command
    with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            subprocess.run(['systemctl', '--user', 'stop', 'b70-qwen38-vllm.service'], check=True, timeout=90)
            with (root/'probe.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
            receipt['status'] = 'PASS_SAMPLER_PROBE'
        except BaseException as error:
            receipt.update(status='FAILED', error=repr(error)); raise
        finally:
            subprocess.run(['docker', 'rm', '-f', 'b70-dflash-sampler-probe'], capture_output=True)
            subprocess.run(['systemctl', '--user', 'start', 'b70-qwen38-vllm.service'], check=True, timeout=90)
            receipt['restoration'] = restore.wait_for_qualified_service(PRODUCTION, timeout=900)
            (root/'campaign.json').write_text(json.dumps(receipt, indent=2)+'\n')
            print(receipt['status'], root, flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Verify that the candidate changes exactly the five pinned Intel packages."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
build = json.loads((ROOT/'runs/build.json').read_text())
probe = r'''
import hashlib, importlib.util, json, subprocess
from pathlib import Path
roots = [Path('/opt/b70'), Path(importlib.util.find_spec('vllm').origin).parent]
roots += list(Path('/opt/venv/lib').glob('python*/site-packages/vllm_xpu_kernels'))
files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
         for root in roots for p in root.rglob('*')
         if p.is_file() and p.suffix in ('.py', '.so')}
assert len(files) > 1000, 'Incomplete inference inventory'
print(json.dumps({'files': files, 'packages': subprocess.check_output([
    'dpkg-query', '-W', '-f=${Package}\t${Version}\n'], text=True)}))
'''
results = {}
for label, image in [('base', build['base']), ('candidate', build['candidate'])]:
    results[label] = json.loads(subprocess.check_output([
        'docker', 'run', '--rm', '--network=none', '--entrypoint', 'python3', image, '-c', probe]))
assert results['base']['files'] == results['candidate']['files'], 'Inference code changed'
base, candidate = ({line.split('\t')[0]: line.split('\t')[1]
                    for line in results[k]['packages'].splitlines()} for k in ['base', 'candidate'])
changed = {k: [base.get(k), candidate.get(k)]
           for k in base.keys() | candidate.keys() if base.get(k) != candidate.get(k)}
assert set(changed) == {'intel-igc-core-2', 'intel-igc-opencl-2', 'intel-ocloc',
                        'intel-opencl-icd', 'libze-intel-gpu1'}, changed
(ROOT/'runs/image-parity.json').write_text(json.dumps({
    'changes': changed, 'inference_files': results['base']['files'], 'passed': True}, indent=2))
print(f'Exactly five Intel packages changed; {len(results["base"]["files"])} inference files match.')

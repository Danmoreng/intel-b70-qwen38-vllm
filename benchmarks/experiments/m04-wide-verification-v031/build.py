#!/usr/bin/env python3
"""Build widened M04 against the exact qualified toolchain, without GPU use."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import argparse

REPO = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument('--name', default='build')
args = parser.parse_args()
ROOT = REPO / 'benchmark-results/m04-wide-verification-v031-20261009' / args.name
BUILDER = 'sha256:a78ff8ea07752bb3bc156cd47043386777031f54531075060889f5b2b55b0505'
KERNEL = '6d92b1bfbf32767ecda8e819613eb151e70030ad'
TLA = '87f6850680a580654b9ea2c80dbc01aeb36ad231'

def archive(repository, commit, destination):
    destination.mkdir(parents=True)
    content = subprocess.check_output(['git', 'archive', commit], cwd=repository)
    subprocess.run(['tar', '-xf', '-', '-C', str(destination)], input=content, check=True)

assert not ROOT.exists()
ROOT.mkdir(parents=True)
archive(REPO.parent / 'b70-xpu-kernels-0.1.15.4-ref', KERNEL, ROOT / 'kernels')
archive(REPO.parent / 'b70-sycl-tla-87f685-reference', TLA, ROOT / 'sycl-tla')
(ROOT / 'output').mkdir()
command = ['docker', 'run', '--rm', '--network', 'none', '--cpus', '4', '--memory', '12g',
           '-v', str(ROOT / 'kernels') + ':/kernels:ro',
           '-v', str(ROOT / 'sycl-tla') + ':/sycl-tla:ro',
           '-v', str(SOURCE) + ':/src:ro', '-v', str(ROOT / 'output') + ':/output',
           '--entrypoint', 'bash', BUILDER, '/src/build-container.sh']
state = dict(status='BUILDING_CPU_ONLY', builder_image=BUILDER, kernel_source_commit=KERNEL,
             sycl_tla_commit=TLA, command=command,
             source_sha256={str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in str(p) and 'build' not in p.relative_to(SOURCE).parts})
(ROOT / 'controller.json').write_text(json.dumps(state, indent=2) + '\n')
try:
    with (ROOT / 'controller.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    manifest = json.loads((ROOT / 'output/manifest.json').read_text())
    assert manifest['status'] == 'COMPILE_AND_CPU_IMPORT_PASS_XPU_UNTESTED'
    state.update(status=manifest['status'], library_sha256=manifest['library_sha256'])
    (SOURCE / 'build').mkdir(exist_ok=False)
    shutil.copyfile(ROOT / 'output/m04.so', SOURCE / 'build/m04.so')
except BaseException as error:
    state.update(status='FAILED', error=repr(error))
    raise
finally:
    (ROOT / 'controller.json').write_text(json.dumps(state, indent=2) + '\n')

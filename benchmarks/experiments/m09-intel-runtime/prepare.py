#!/usr/bin/env python3
"""Freeze the completed M04 workload for a candidate-only Intel runtime rerun."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BASELINE = REPO / "benchmarks/experiments/m06-target-head-int4/runs/coding-20260919-101848"


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def main():
    original = json.loads((BASELINE / 'manifest.json').read_text())
    for name, sha in original['harness_sha256'].items():
        if digest(BASELINE / 'harness' / name) != sha:
            raise RuntimeError(f'Original harness changed: {name}')
    if digest(BASELINE / 'source.tar') != original['source_sha256']:
        raise RuntimeError('Original source archive changed')
    with tarfile.open(BASELINE/'source.tar') as archive:
        for member in archive:
            if member.isfile():
                expected = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                if digest(BASELINE/'frozen'/member.name) != expected:
                    raise RuntimeError(f'Frozen source differs from archive: {member.name}')
    build = json.loads((ROOT/'runs/build.json').read_text())
    agent = Path(original['agent_runtime']) / '@earendil-works/pi-coding-agent/package.json'
    if json.loads(agent.read_text())['version'] != original['agent_version']:
        raise RuntimeError('Agent version changed')
    run = ROOT / 'runs' / datetime.datetime.now().strftime('prepared-%Y%m%d-%H%M%S')
    run.mkdir(parents=True)
    for name in ['frozen', 'dependencies', 'harness']:
        # Reflinks where supported; never share writable hardlinks with the reference.
        subprocess.run(['cp', '-a', '--reflink=auto', str(BASELINE/name), str(run/name)], check=True)
    shutil.copy2(BASELINE/'source.tar', run/'source.tar')
    boundary = Path('/home/sebastian/LocalLLM/Local-AI-B70/qwen38/optimization/q128-full-context-208k/runs/run-20260914-215108/request.json')
    shutil.copy2(boundary, run/'context-boundary-request.json')
    shutil.copy2(BASELINE/'summary.json', run/'reference-summary.json')
    shutil.copy2(BASELINE/'arm-0-control/task-result.json', run/'reference-task-result.json')
    shutil.copy2(BASELINE/'manifest.json', run/'reference-manifest.json')
    manifest = {**original, 'candidate': build['candidate'], 'runtime_build': build,
        'status': 'prepared-awaiting-serving-gates',
        'reference_run': str(BASELINE), 'reference_arm': 'arm-0-control',
        'comparison': 'Same frozen coding tasks, inference settings and limits; new NEO 26.35.39758.10 / IGC 2.41.5 versus historical production reference. Adaptive trajectories may differ.',
        'boundary_request_sha256': digest(run/'context-boundary-request.json'),
        'source_files': hashes(run/'frozen'),
        'harness_files': hashes(run/'harness'),
        'dependency_files': hashes(run/'dependencies'),
        'preflight_required': ['startup-200704', 'nonzero-draft-acceptance',
                               'prefix-state-correctness', 'full-context-200448-plus-256',
                               'isolated-agent-tool-smoke'],
    }
    (run/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (run/'preflight.json').write_text(json.dumps({'ready': False, 'reason': 'Serving gates not run'},indent=2)+'\n')
    (ROOT/'LATEST').write_text(str(run)+'\n')
    print(run)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Freeze the completed M04 workload for a candidate-only DFlash2 rerun."""
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
DRAFT = Path('/home/sebastian/LocalLLM/models-legacy/Qwen3.8-27B-DFlash2-BF16')


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
    expected_draft = '67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c'
    if digest(DRAFT / 'model.safetensors') != expected_draft:
        raise RuntimeError('Draft checkpoint changed')
    agent = Path(original['agent_runtime']) / '@earendil-works/pi-coding-agent/package.json'
    if json.loads(agent.read_text())['version'] != original['agent_version']:
        raise RuntimeError('Agent version changed')
    run = ROOT / 'runs' / datetime.datetime.now().strftime('prepared-%Y%m%d-%H%M%S')
    run.mkdir(parents=True)
    for name in ['frozen', 'dependencies', 'harness']:
        # Reflinks where supported; never share writable hardlinks with the reference.
        subprocess.run(['cp', '-a', '--reflink=auto', str(BASELINE/name), str(run/name)], check=True)
    shutil.copy2(BASELINE/'source.tar', run/'source.tar')
    shutil.copy2(BASELINE/'summary.json', run/'reference-summary.json')
    shutil.copy2(BASELINE/'arm-0-control/task-result.json', run/'reference-task-result.json')
    shutil.copy2(BASELINE/'manifest.json', run/'reference-manifest.json')
    manifest = {**original, 'candidate': original['control'],
        'status': 'prepared-awaiting-serving-gates',
        'reference_run': str(BASELINE), 'reference_arm': 'arm-0-control',
        'draft_path': str(DRAFT), 'draft_sha256': expected_draft,
        'draft_config_sha256': digest(DRAFT/'config.json'),
        'draft_revision': 'dedf8df68adfb1afeaf7b7480c0a0243108177b4',
        'candidate_dtype': 'bfloat16',
        'candidate_spec': {'method': 'dflash', 'model': '/draft', 'num_speculative_tokens': 7},
        'comparison': 'Same frozen coding tasks and limits; historical M04 reference, adaptive trajectory. DFlash candidate changes speculation AND activation dtype.',
        'source_files': hashes(run/'frozen'),
        'harness_files': hashes(run/'harness'),
        'dependency_files': hashes(run/'dependencies'),
        'preflight_required': ['startup-200704', 'nonzero-draft-acceptance',
                               'prefix-state-correctness', 'long-context-143034',
                               'isolated-agent-tool-smoke'],
    }
    (run/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (run/'preflight.json').write_text(json.dumps({'ready': False, 'reason': 'Serving gates not run'},indent=2)+'\n')
    (ROOT/'LATEST').write_text(str(run)+'\n')
    print(run)


if __name__ == '__main__':
    main()

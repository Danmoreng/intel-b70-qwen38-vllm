#!/usr/bin/env python3
"""Build a direct-XPU candidate from pinned public sources, without the GPU.

This does not stop services, change production manifests, or publish anything.
"""
import argparse
from datetime import datetime
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import urllib.request
from zoneinfo import ZoneInfo

from release_integrity import check_build_tag

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / 'engine/direct-xpu'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_onednn_source_version(source, record):
    # Release tags and embedded versions can differ upstream. Check the
    # explicitly recorded library version before spending time compiling.
    text = (source / 'CMakeLists.txt').read_text()
    match = re.search(r'set\(\s*PROJECT_VERSION\s+"([^"]+)"\s*\)', text)
    if not match or match[1] != record['runtime_version']:
        raise RuntimeError('oneDNN source version differs from lock: ' +
                           (match[1] if match else 'missing PROJECT_VERSION'))


def prepare_context(out, download_cache=None):
    lock = json.loads((SOURCE / 'build-lock.json').read_text())
    manifest = json.loads((SOURCE / 'source-manifest.json').read_text())
    actual_files = {str(p.relative_to(SOURCE)) for folder in ('runtime', 'vllm', 'm04', 'pti')
                    for p in (SOURCE / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual_files != set(manifest['files_sha256']):
        raise RuntimeError('Integration source inventory differs from its manifest')
    for name, expected in manifest['files_sha256'].items():
        if sha(SOURCE / name) != expected:
            raise RuntimeError('Integration source manifest mismatch: ' + name)
    requirements = ''.join(f'{name}=={version}\n' for name, version in lock['python_packages'].items())
    if (SOURCE / 'requirements.txt').read_text() != requirements:
        raise RuntimeError('Requirements and build lock differ')
    context = out / 'context'
    context.mkdir()
    shutil.copytree(SOURCE, context / 'direct', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(REPO / 'engine/exl3xpu/mb4-k4/source', context / 'matrix-source')
    m04 = context / 'm04-source'
    m04.mkdir()
    for name in ('csrc', 'patches'):
        shutil.copytree(SOURCE / 'm04' / name, m04 / name)
    shutil.copytree(REPO / 'runtime', context / 'runtime', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(REPO / 'engine/cache1792/cow_copy_fix.py', context / 'cow_copy_fix.py')
    shutil.copyfile(REPO / 'docker/patches/b70_mtp_policy.py', context / 'b70_mtp_policy.py')
    shutil.copyfile(REPO / 'docker/direct-xpu/Dockerfile', context / 'Dockerfile')
    if lock['upstream_image'] not in (context / 'Dockerfile').read_text():
        raise RuntimeError('Dockerfile and locked upstream image differ')
    downloads = out / 'downloads'
    downloads.mkdir()
    for record in lock.get('artifacts', []):
        filename = record['filename']
        if Path(filename).name != filename:
            raise RuntimeError('Invalid artifact filename')
        artifact = downloads / filename
        cached = download_cache / filename if download_cache else None
        if cached and cached.is_file():
            shutil.copyfile(cached, artifact)
        else:
            with urllib.request.urlopen(record['url'], timeout=90) as response, artifact.open('wb') as dest:
                shutil.copyfileobj(response, dest)
        if sha(artifact) != record['sha256']:
            raise RuntimeError('Artifact checksum mismatch: ' + filename)
        (context / 'artifacts').mkdir(exist_ok=True)
        shutil.copyfile(artifact, context / 'artifacts' / filename)
    for name, record in lock['sources'].items():
        archive = downloads / (name + '.tar.gz')
        with urllib.request.urlopen(record['url'], timeout=90) as response, archive.open('wb') as dest:
            shutil.copyfileobj(response, dest)
        if sha(archive) != record['sha256']:
            raise RuntimeError('Source archive checksum mismatch: ' + name)
        unpack = out / ('unpack-' + name)
        unpack.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(unpack, filter='data')
        children = list(unpack.iterdir())
        if len(children) != 1 or not children[0].is_dir():
            raise RuntimeError('Unexpected source archive layout: ' + name)
        (context / 'sources').mkdir(exist_ok=True)
        shutil.move(str(children[0]), context / 'sources' / name)
        unpack.rmdir()
        if name == 'onednn':
            check_onednn_source_version(context / 'sources' / name, record)
    inventory = {str(p.relative_to(context)): sha(p) for p in sorted(context.rglob('*')) if p.is_file()}
    (out / 'context-sha256.json').write_text(json.dumps(inventory, indent=2) + '\n')
    return context, lock


def cleanup_build_containers(log):
    """Remove only intermediate IDs emitted by this build's Docker client."""
    if not log.exists():
        return []
    ids = re.findall(r'^ ---> Running in ([0-9a-f]{12,64})$', log.read_text(), re.M)
    removed = []
    for container in dict.fromkeys(ids):
        if subprocess.run(['docker', 'container', 'inspect', container], capture_output=True, timeout=10).returncode == 0:
            subprocess.run(['docker', 'rm', '-f', container], check=True, capture_output=True, timeout=30)
            removed.append(container)
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prepare-only', action='store_true', help='Download and verify the build context, without Docker')
    parser.add_argument('--download-cache', type=Path, help='Optional existing wheel cache; checksums are always verified')
    parser.add_argument('--timeout-seconds', type=int, default=10800)
    args = parser.parse_args()
    check_build_tag(args.tag, REPO)
    if not 60 <= args.timeout_seconds <= 10800:
        raise SystemExit('Build deadline must be between 60 seconds and three hours')
    now = datetime.now(ZoneInfo('Europe/Berlin'))
    available = (now.replace(hour=22, minute=45, second=0, microsecond=0) - now).total_seconds()
    if available < args.timeout_seconds + 600:
        raise SystemExit('Insufficient time before the daily closing phase at 22:45 Europe/Berlin')
    if not args.prepare_only and subprocess.run(['docker', 'image', 'inspect', args.tag], capture_output=True).returncode == 0:
        raise SystemExit('Refusing to replace an existing image tag')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'PREPARING', 'tag': args.tag, 'started_at': now.isoformat(),
             'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
             'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()),
             'gpu_tests_executed': False, 'production_changed': False}
    def save():
        (out / 'build.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    try:
        context, lock = prepare_context(out, args.download_cache)
        state.update(status='CONTEXT_READY', build_lock=lock, context_manifest_sha256=sha(out / 'context-sha256.json'))
        save()
        if args.prepare_only:
            print(json.dumps({'status': state['status'], 'context': str(context)}))
            return
        # Downloads also consume the available daytime window. Recheck before
        # Docker so preparation cannot silently push a long build past closing.
        build_now = datetime.now(ZoneInfo('Europe/Berlin'))
        remaining = (build_now.replace(hour=22, minute=45, second=0, microsecond=0) - build_now).total_seconds() - 600
        if remaining < 60:
            raise RuntimeError('Preparation exhausted the safe build window')
        build_timeout = min(args.timeout_seconds, int(remaining))
        command = ['docker', 'build', '--force-rm', '--memory', '6g', '--memory-swap', '6g',
                   '--cpu-quota', '400000', '--iidfile', str(out / 'image-id'), '-t', args.tag, str(context)]
        state.update(status='BUILDING', command=command, effective_build_timeout_seconds=build_timeout)
        save()
        with (out / 'build.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=build_timeout)
        state.update(status='BUILT_CPU_CHECKED_GPU_UNTESTED', image_id=(out / 'image-id').read_text().strip())
    except BaseException as exc:
        state.update(status='FAILED', error=repr(exc))
        try:
            state['removed_build_containers'] = cleanup_build_containers(out / 'build.log')
        except Exception as cleanup_error:
            state['cleanup_error'] = repr(cleanup_error)
        raise
    finally:
        state['finished_unix'] = time.time()
        save()
    print(json.dumps({'status': state['status'], 'image_id': state['image_id'], 'receipt': str(out / 'build.json')}))


if __name__ == '__main__':
    main()

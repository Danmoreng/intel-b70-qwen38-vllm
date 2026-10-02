#!/usr/bin/env python3
"""Publish the tracked source of the already qualified EXL3 v2 release."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exl3-repo', type=Path, required=True)
    args = parser.parse_args()
    release = json.loads((REPO/'config/production_image.json').read_text())
    assert release['status'] == 'QUALIFIED_RELEASE'
    receipt = release['production_start_receipt']
    assert sha(REPO/receipt['path']) == receipt['sha256']
    assert json.loads((REPO/receipt['path']).read_text())['status'] == 'PASS_STRICT_PRODUCTION_DEPLOYMENT_AND_RESTART'
    source = REPO/'engine/exl3xpu'
    candidate = source/'review-candidate'
    manifest = json.loads((candidate/'manifest.json').read_text())
    build = json.loads((candidate/'build-receipt.json').read_text())
    assert manifest['source_commit'] == release['source_commit']
    assert build['image_id'] == release['runtime_image_id']
    assert build['native_manifest']['library_sha256'] == release['runtime_artifacts_sha256']['/opt/exl3xpu/exl3xpu/_C.so']
    for name, digest in manifest['files_sha256'].items():
        assert sha(candidate/name) == digest
    with tarfile.open(candidate/'source.tar.gz', 'r:gz') as archive:
        for name, digest in build['source_sha256'].items():
            member = archive.extractfile('exl3xpu/'+name)
            assert member is not None and hashlib.sha256(member.read()).hexdigest() == digest, name
    old = json.loads((REPO/'config/releases/exl3-v1/source-snapshot/manifest.json').read_text())
    patch = subprocess.check_output(['git', '-C', str(args.exl3_repo), 'diff', '--binary', old['base_commit'], release['source_commit']])
    shutil.copyfile(candidate/'source.tar.gz', source/'source.tar.gz')
    (source/'migration.patch').write_bytes(patch)
    published = dict(upstream=old['upstream'], base_commit=old['base_commit'], source_commit=release['source_commit'],
        source_commit_note='Complete tracked source; all 74 build receipt inputs verified against the archive. EXL3 v1 snapshot retained separately.',
        runtime_image=release['runtime_image_id'], production_image=release['image_id'],
        source_sha256=build['source_sha256'], native_manifest=build['native_manifest'], m04_manifest=build['m04_manifest'],
        build_receipt=dict(path='engine/exl3xpu/review-candidate/build-receipt.json', sha256=sha(candidate/'build-receipt.json')),
        qualification_review=release['qualification_review'],
        files_sha256={name:sha(source/name) for name in ['source.tar.gz', 'migration.patch']})
    (source/'manifest.json').write_text(json.dumps(published, indent=2)+'\n')
    (source/'README.md').write_text('''# Qualified EXL3 v2 engine source

This directory publishes the complete tracked EXL3 source and upstream-to-release
patch in the user-owned repository. The manifest pins the upstream base, source
commit, all 74 build inputs, compiler/native identities and qualified image.
Native binaries, weights, caches and large numerical arrays are omitted. The
upstream license is preserved inside `source.tar.gz`.

Extract `source.tar.gz`, or check out the upstream/base and apply `migration.patch`.
The runtime parent and final metadata-only release child have identical filesystem
layers. The child explicitly sets the partition-cache capacity to 64. The
historical `review-candidate` directory preserves the pre-qualification build
receipt; its old candidate status is not the current release decision.

Build with `scripts/build_target_image.py --tag <fresh-tag> --output <fresh-receipt-directory>
--gptq-repo <this-project> --m04-artifacts <verified-m04-directory> --onednn <oneDNN3.13-install>`.
The source pins builder/base identities. Native recompilation uses Torch 2.13 /
SYCL 9; `--native-artifacts` requires source/hash/ABI-verified reuse. M04 source
and build recipe remain in `benchmarks/experiments/exl3-shared-kv-verify`.

A build is not release approval. The [v2 release report](../../docs/EXL3_RELEASE_V2_REPORT.md)
links the actual serving, numerical, operating and deployment gates. Use a fresh
build tag, never overwrite a qualified production or rollback alias.
The [saved v1 source and runtime](../../config/releases/exl3-v1/README.md) remain
available for immediate rollback. No changes were pushed to 0xSero upstream.
''')
    print(json.dumps(dict(status='PASS_QUALIFIED_SOURCE_PUBLICATION', source_commit=release['source_commit'], build_inputs=len(build['source_sha256']), files_sha256=published['files_sha256']), indent=2))


if __name__ == '__main__':
    main()

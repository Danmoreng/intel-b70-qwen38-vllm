#!/usr/bin/env python3
"""Publish the tracked source of the already qualified EXL3 release."""
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
    parser.add_argument('--candidate-dir',type=Path,default=REPO/'engine/exl3xpu/review-candidate')
    args = parser.parse_args()
    release = json.loads((REPO/'config/production_image.json').read_text())
    assert release['status'] == 'QUALIFIED_RELEASE'
    receipt = release['production_start_receipt']
    assert sha(REPO/receipt['path']) == receipt['sha256']
    assert json.loads((REPO/receipt['path']).read_text())['status'] == 'PASS_STRICT_PRODUCTION_DEPLOYMENT_AND_RESTART'
    source = REPO/'engine/exl3xpu'
    candidate = args.candidate_dir.resolve()
    candidate_path = str(candidate.relative_to(REPO))
    policy = json.loads((REPO/'config/production_policy.json').read_text())
    version = policy['policy_id'].rsplit('-',1)[-1]
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
        source_commit_note=f"Complete tracked source; all {len(build['source_sha256'])} build inputs verified against the archive. Prior EXL3 source/runtime snapshots retained separately.",
        runtime_image=release['runtime_image_id'], production_image=release['image_id'],
        source_sha256=build['source_sha256'], native_manifest=build['native_manifest'], m04_manifest=build['m04_manifest'],
        build_receipt=dict(path=candidate_path+'/build-receipt.json', sha256=sha(candidate/'build-receipt.json')),
        qualification_review=release['qualification_review'],
        production_start_receipt=release['production_start_receipt'],
        files_sha256={name:sha(source/name) for name in ['source.tar.gz', 'migration.patch']})
    attestation=candidate/'m04-controller-attestation.json'
    if attestation.exists():
        assert manifest['files_sha256']['m04-controller-attestation.json']==sha(attestation)
        published['m04_controller_attestation']=dict(path=candidate_path+'/m04-controller-attestation.json',sha256=sha(attestation))
    if policy['runtime']['vllm']=='0.31.0':
        requirements=REPO/'config/experiments/vllm-031-upgrade/runtime_requirements.json'
        published['runtime_requirements']=dict(path=str(requirements.relative_to(REPO)),sha256=sha(requirements))
    (source/'manifest.json').write_text(json.dumps(published, indent=2)+'\n')
    (source/'README.md').write_text(f"""# Qualified EXL3 {version} engine source

The complete source archive preserves the upstream license. The manifest pins
upstream/base, source commit, all {len(build['source_sha256'])} build inputs,
native compiler/ABI identities and the actual qualified production image.
Weights, native binaries, caches and large numerical arrays are omitted.
Prior v1/v2 configuration and source snapshots remain under `config/releases`.

Extract `source.tar.gz`, or apply `migration.patch` to the recorded upstream
base. The release child and runtime parent have identical filesystem layers.
The native partition-cache capacity is 64 per inference thread/queue.

Current toolchain: vLLM {policy['runtime']['vllm']},
Torch {policy['runtime']['torch']}, oneAPI {policy['runtime']['oneapi_compiler']},
oneDNN {policy['runtime']['onednn']}, C++20 and SYCL 9 for the 0.31 update.
Runtime distribution pins are in `config/experiments/vllm-031-upgrade/runtime_requirements.json`.
The default `scripts/build-image.sh` builds the pinned vLLM 0.31 XPU compiler/base;
`docker/Dockerfile.vllm030-gptq-legacy` preserves the historical GPTQ base recipe.

From the extracted source, build a fresh candidate with:

```bash
python3 scripts/build_target_image.py --tag <fresh-candidate-tag> \\
  --output <fresh-receipt-directory> --builder-image <pinned-builder-image> \\
  --base-image <pinned-base-image> --gptq-repo <this-project> \\
  --m04-artifacts <verified-Torch-2.14-M04-directory> \\
  --onednn <oneDNN-3.13-install>
```

`--native-artifacts` permits only source/hash/ABI-verified native reuse.
M04 source remains in `benchmarks/experiments/exl3-shared-kv-verify`; the executed
Torch 2.14 wrapper and qualification controllers are in
`benchmarks/experiments/vllm-031-upgrade`. Native libraries must be rebuilt for
the new Torch ABI. A build alone does not approve deployment: the release
manifest links the actual gate assessment and strict service startup receipt.
No changes were pushed to 0xSero upstream.
""")
    print(json.dumps(dict(status='PASS_QUALIFIED_SOURCE_PUBLICATION', source_commit=release['source_commit'], build_inputs=len(build['source_sha256']), files_sha256=published['files_sha256']), indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Export the existing qualified receipts; performs no GPU work."""
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[3]
PUBLIC = REPO / 'benchmarks/results/vllm-031-upgrade'
LOCAL = REPO / 'benchmark-results/vllm-031-upgrade-20261009'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    release = json.loads((REPO / 'config/production_image.json').read_text())
    assert release['status'] == 'QUALIFIED_RELEASE'
    assessment = json.loads((PUBLIC / 'assessment.json').read_text())
    assert assessment['image_id'] == release['image_id']
    assert assessment['status'] == 'PASS_EXACT_IMAGE_RELEASE_GATES'
    promotion = json.loads((PUBLIC / 'promotion.json').read_text())
    assert promotion['status'] == 'PASS_STRICT_PRODUCTION_DEPLOYMENT_AND_RESTART'
    assert sha(PUBLIC / 'promotion.json') == release['production_start_receipt']['sha256']
    receipts = dict(assessment['receipts_sha256'])
    measured = json.loads((PUBLIC / 'performance-review-measurements.json').read_text())
    for item in measured['receipts'].values():
        receipts[item['path']] = item['sha256']
    for name in [
        'source-guard-review.json', 'trace-port-attestation.json',
        'final-numerics-controller.json', 'final-serving-controller.json',
        'quality-final/campaign.json', 'quality-attempt2/campaign.json',
        'quality-attempt2/quality-check.log', 'operations-final/runtime-sources.zip',
        'metadata-image-r2/image.json', 'v2-rollback-after-upgrade-preflight.log'
    ]:
        path = LOCAL / name
        receipts[str(path.relative_to(REPO))] = sha(path)
    serving = json.loads((LOCAL / 'final-serving-controller.json').read_text())
    for name, digest in serving['sources_sha256'].items():
        assert sha(REPO / name) == digest
        receipts[name] = digest
    mapping = {}
    for name, digest in receipts.items():
        original = REPO / name
        assert sha(original) == digest, name
        assert original.stat().st_size < 2 * 1024 * 1024, name
        if name.startswith('benchmark-results/'):
            target = PUBLIC / 'evidence' / name.removeprefix('benchmark-results/')
            if target.suffix == '.log':
                target = target.with_suffix('.txt')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert sha(target) == digest
            else:
                shutil.copyfile(original, target)
        else:
            target = original
        mapping[name] = dict(public_path=str(target.relative_to(REPO)), sha256=digest)
    manifest = dict(
        status='PASS_QUALIFIED_PUBLICATION_EXPORT', image_id=release['image_id'],
        source_receipts=mapping,
        files_sha256={str(p.relative_to(PUBLIC)): sha(p) for p in sorted(PUBLIC.rglob('*'))
                      if p.is_file() and p.name != 'PUBLICATION_MANIFEST.json'},
        scope='Existing measured evidence exported byte-for-byte. Original local paths in signed receipts map to public paths above. Full token-counter results, original-tolerance state replays, active source coverage and licensed runtime sources are included. Model weights, native binaries, logits arrays, raw SSE and large prompt fixtures remain local. Historical baseline and power campaigns were not rerun. Failed old-runtime bitidentity check remains visible; numerical and performance tradeoffs have explicit reviews.')
    (PUBLIC / 'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(status=manifest['status'], receipts=len(mapping), exported_files=len(manifest['files_sha256'])), indent=2))


if __name__ == '__main__':
    main()

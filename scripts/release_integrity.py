"""Read-only operational-helper checks, before any side effects."""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVICE = 'b70-qwen38-vllm.service'
CONTAINER = 'b70-qwen38-vllm'


def load_release(repo=REPO):
    directory = Path(repo) / 'config'
    policy = directory / 'production_policy.json'
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    if (directory / 'production_policy.sha256').read_text().split()[0] != digest:
        raise RuntimeError('Production policy hash mismatch')
    release = json.loads((directory / 'production_image.json').read_text())
    if release['policy_sha256'] != digest or release['status'] != 'QUALIFIED_RELEASE':
        raise RuntimeError('Current release is not qualified or policy identity changed')
    for key in ('qualification_review', 'production_start_receipt'):
        receipt = release[key]
        path = Path(repo) / receipt['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['sha256']:
            raise RuntimeError('Release receipt changed: ' + key)
    return release


def check_build_tag(tag, repo=REPO):
    directory = Path(repo) / 'config'
    manifests = [directory / 'production_image.json']
    manifests += sorted((directory / 'releases').glob('*/production_image.json'))
    protected = {json.loads(path.read_text())['image_tag'] for path in manifests}
    if (not tag or tag in protected or '@' in tag or tag.startswith('sha256:') or
            tag.rsplit(':', 1)[-1].startswith('production-')):
        raise ValueError('Base builds require a separate development/base tag; protected release tag: ' + tag)
    return tag


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-build-tag', required=True)
    args = parser.parse_args()
    try:
        check_build_tag(args.check_build_tag)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        raise SystemExit(str(exc))

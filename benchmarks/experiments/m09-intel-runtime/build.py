#!/usr/bin/env python3
"""Add only checksum-pinned NEO/IGC packages to the production image."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent
BASE = 'sha256:aee9857bef1f37c8f0ee136d9f89d7166201212175a8b171d958627706cf1c0b'
TAG = 'local/qwen38-b70-vllm:runtime-26.35-igc-2.41.5-20260919'
PACKAGES = [
    ('intel/intel-graphics-compiler', 'v2.41.5', 'intel-igc-core-2_2.41.5+22716_amd64.deb', '0a6e64a663ae65a0fa02d6912ae3b6b37cf85b90c21cc423fd9fef70aaf4f628'),
    ('intel/intel-graphics-compiler', 'v2.41.5', 'intel-igc-opencl-2_2.41.5+22716_amd64.deb', '779e1b9e88098eb25711e9a8f67c2752665bad22f134aa40ed5649f6e1b87058'),
    ('intel/compute-runtime', '26.35.39758.10', 'intel-ocloc_26.35.39758.10-0_amd64.deb', 'c64bff586edf2bd9b49f3e4c9c2be25e05c43466e4dafa2a2c3948d498c736b7'),
    ('intel/compute-runtime', '26.35.39758.10', 'intel-opencl-icd_26.35.39758.10-0_amd64.deb', '61712caaddeba3d38e4f79e2a0fb23fea25596ca2d72c3144c6eea2331ec4301'),
    ('intel/compute-runtime', '26.35.39758.10', 'libze-intel-gpu1_26.35.39758.10-0_amd64.deb', 'c19a641b953d55aebbf1d51bec364a84bf629f985e02fbbe6dc70224c0e88470'),
]


def main():
    context = ROOT/'runs/build'
    context.mkdir(parents=True, exist_ok=True)
    for repo, version, name, sha in PACKAGES:
        path = context/name
        if not path.exists():
            urllib.request.urlretrieve(f'https://github.com/{repo}/releases/download/{version}/{name}', path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise RuntimeError(f'Checksum mismatch: {name}')
        subprocess.run(['docker', 'run', '--rm', '--network=none', '-v', f'{context}:/packages:ro',
                        '--entrypoint', 'dpkg-deb', BASE, '-f', f'/packages/{name}',
                        'Package', 'Version', 'Depends'], check=True)
    (context/'Dockerfile').write_text(
        f'FROM {BASE}\nCOPY *.deb /tmp/intel-runtime/\n'
        'RUN dpkg -i /tmp/intel-runtime/*.deb && rm -rf /tmp/intel-runtime && ldconfig\n')
    subprocess.run(['docker', 'build', '--network=none', '-t', TAG, str(context)], check=True)
    inspect = lambda image: json.loads(subprocess.check_output(['docker', 'image', 'inspect', image]))[0]
    before, after = inspect(BASE), inspect(TAG)
    if after['RootFS']['Layers'][:len(before['RootFS']['Layers'])] != before['RootFS']['Layers']:
        raise RuntimeError('Candidate is not an additive production image')
    for key in ['Env', 'Entrypoint', 'Cmd', 'WorkingDir', 'User']:
        if before['Config'].get(key) != after['Config'].get(key):
            raise RuntimeError(f'Image configuration changed: {key}')
    result = {'base': BASE, 'candidate': after['Id'], 'tag': TAG, 'packages': PACKAGES}
    (ROOT/'runs/build.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

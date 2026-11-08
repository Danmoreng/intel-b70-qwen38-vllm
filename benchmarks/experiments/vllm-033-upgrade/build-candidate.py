#!/usr/bin/env python3
"""Reproduce the Python-only 0.33 port on the qualified native 0.31 image."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from zoneinfo import ZoneInfo

EXP = Path(__file__).resolve().parent
REPO = EXP.parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
from release_integrity import check_build_tag


def inspect(image):
    return json.loads(subprocess.check_output(['docker', 'image', 'inspect', image], text=True))[0]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wheel', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--tag', required=True)
    p.add_argument('--base-image', default='sha256:36e41ea027758060f9352bdd7f9ed47de5143d0424b238653960dbf8357c176a',
                   help='Native0.31 base; alternative source rebuilds require their own complete qualification')
    a = p.parse_args()
    if dt.datetime.now(ZoneInfo('Europe/Berlin')).hour >= 22:
        raise SystemExit('Start a bounded build earlier than22:00 Europe/Berlin')
    check_build_tag(a.tag)
    for tag in (a.tag, a.tag + '-deps'):
        if subprocess.run(['docker', 'image', 'inspect', tag], capture_output=True).returncode == 0:
            raise RuntimeError('Refusing to replace existing tag: ' + tag)
    base = inspect(a.base_image)['Id']
    if base != 'sha256:36e41ea027758060f9352bdd7f9ed47de5143d0424b238653960dbf8357c176a':
        # A source rebuild has a different image identity. Validate its native
        # provenance and ABI before using it as an experimental parent; the
        # resulting image does not inherit this host's qualification.
        code = "import runpy,torch,vllm; assert torch.__version__=='2.14.0+xpu'; assert vllm.__version__=='0.31.0'; runpy.run_path('/opt/exl3xpu/scripts/verify_native_artifact.py')"
        subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'python',
                        base, '-c', code], check=True, timeout=120)
    assert sha(a.wheel) == '0e3735aa6b9ac132a32b54a948bb0de079e1b783ec5b04b184f4bdaf873eb995'
    archive = REPO / 'engine/exl3xpu/source.tar.gz'
    assert sha(archive) == 'c5b46a9af81ca41bf04a95b9bf4a48044b5bad3654cb61fe327af857ae93da43'
    root = a.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as source:
        source.extractall(root, filter='data')
    subprocess.run([sys.executable, str(EXP / 'port-source.py'), '--source', str(root / 'exl3xpu'),
                    '--wheel', str(a.wheel.resolve()), '--stock', str(root / 'stock'),
                    '--policy', str(REPO / 'docker/patches/b70_mtp_policy.py')], check=True, timeout=120)
    deps = root / 'deps'
    deps.mkdir()
    shutil.copyfile(a.wheel, deps / 'vllm-0.33.0+xpu-cp38-abi3-manylinux_2_34_x86_64.whl')
    base_recipe = (EXP / 'Dockerfile.pinned').read_text().splitlines(True)
    base_recipe[0] = 'FROM ' + base + '\n'
    (deps / 'Dockerfile').write_text(''.join(base_recipe))
    with (root / 'deps-build.log').open('w') as log:
        subprocess.run(['docker', 'build', '-t', a.tag + '-deps', str(deps)],
                       check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    deps_image = inspect(a.tag + '-deps')['Id']
    port = root / 'port'
    (port / 'exl3xpu').mkdir(parents=True)
    (port / 'tests').mkdir()
    for name in ('target_runtime.py', 'attention_dispatch.py', 'gdn_target_patch.py', 'vllm_plugin.py'):
        shutil.copyfile(root / 'exl3xpu/exl3xpu' / name, port / 'exl3xpu' / name)
    shutil.copyfile(root / 'exl3xpu/tests/test_target_gdn.py', port / 'tests/test_target_gdn.py')
    for name in ('qwen3_5_mtp.py', 'b70_mtp_policy.py'):
        shutil.copyfile(root / 'stock/vllm/model_executor/models' / name, port / name)
    shutil.copyfile(root / 'exl3xpu/models/qwen3.8-27b-exl3-4.00bpw/migration-target-optimized.yaml',
                    port / 'migration-target-optimized.yaml')
    text = (EXP / 'Dockerfile.port').read_text().splitlines(True)
    text[0] = 'FROM ' + deps_image + '\n'
    (port / 'Dockerfile').write_text(''.join(text))
    with (root / 'port-build.log').open('w') as log:
        subprocess.run(['docker', 'build', '--network', 'none', '-t', a.tag, str(port)],
                       check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    receipt = {'status': 'BUILT_NOT_QUALIFIED', 'base_image_id': base, 'deps_image_id': deps_image,
               'candidate_image_id': inspect(a.tag)['Id'], 'candidate_tag': a.tag,
               'wheel_sha256': sha(a.wheel), 'source_archive_sha256': sha(archive)}
    (root / 'image.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

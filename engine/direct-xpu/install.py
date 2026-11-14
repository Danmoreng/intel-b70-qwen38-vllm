"""Install the audited final 0.33 integration and bind freshly built artifacts.

Only known upstream boundaries are accepted. This never approves a release.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import yaml

ROOT = Path('/opt/b70-direct')
ENGINE = Path('/opt/exl3xpu')
VLLM = Path('/opt/venv/lib/python3.12/site-packages/vllm')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    lock = json.loads((ROOT / 'build-lock.json').read_text())
    sources = json.loads((ROOT / 'source-manifest.json').read_text())
    for name, expected in sources['files_sha256'].items():
        if name.startswith('runtime/'):
            actual = ENGINE / name.removeprefix('runtime/')
        elif name.startswith('m04/'):
            actual = ENGINE / name
        else:
            actual = ROOT / name
        if sha(actual) != expected:
            raise RuntimeError('Unreviewed integration source: ' + name)
    torch_init = VLLM.parent / 'torch/__init__.py'
    if sha(torch_init) != lock['pristine_torch_init_sha256']:
        raise RuntimeError('Unexpected modification to official Torch loader')
    pti_receipt = json.loads(Path('/native/pti-patch.json').read_text())
    assert pti_receipt['base_commit'] == lock['sources']['pti']['commit']
    assert pti_receipt['fix_commit'] == lock['sources']['pti']['fix_commit']
    for path, expected in pti_receipt['library_sha256'].items():
        assert sha(path) == expected, path
    (ROOT / 'pti-patch.json').write_text(json.dumps(pti_receipt, indent=2) + '\n')
    for name, expected in lock['pristine_vllm_sha256'].items():
        if sha(VLLM / name) != expected:
            raise RuntimeError('Unsupported upstream vLLM boundary: ' + name)
    for name in lock['pristine_vllm_sha256']:
        shutil.copyfile(ROOT / 'vllm' / name, VLLM / name)
    shutil.copyfile(ROOT / 'b70_mtp_policy.py', VLLM / 'model_executor/models/b70_mtp_policy.py')
    shutil.copyfile(ROOT / 'b70_cow_copy_fix.py', VLLM / 'v1/worker/b70_cow_copy_fix.py')
    for source, dest in [
        ('_C.so', 'exl3xpu/_C.so'), ('_C.so.manifest.json', 'exl3xpu/_C.so.manifest.json'),
        ('m04.so', 'm04/m04.so'), ('m04.manifest.json', 'm04/manifest.json'),
    ]:
        shutil.copyfile(Path('/native') / source, ENGINE / dest)
    # Preserve the guard's implementation; bind its expected artifact identities
    # to this build. The plugin remains byte-identical to the reference source.
    guard = ENGINE / 'exl3xpu/target_matrix_guard.py'
    text = guard.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'EXPECTED')
    previous = ast.literal_eval(node.value)
    expected = {name: sha(ENGINE / 'exl3xpu' / name) for name in previous}
    lines = text.splitlines(True)
    lines[node.lineno - 1:node.end_lineno] = ['EXPECTED = ' + repr(expected) + '\n']
    guard.write_text(''.join(lines))
    profile = ENGINE / 'models/qwen3.8-27b-exl3-4.00bpw/migration-target-optimized.yaml'
    text = profile.read_text()
    original = yaml.safe_load(text)['env']['EXL3_M04_LIBRARY_SHA256']
    assert text.count(original) == 1
    profile.write_text(text.replace(original, sha(ENGINE / 'm04/m04.so')))
    (ROOT / 'installation.json').write_text(json.dumps({
        'status': 'INSTALLED_BUILD_NOT_QUALIFIED',
        'reference_source_image': sources['source_image'],
        'native_guard_previous': previous, 'native_guard_current': expected,
        'installed_guard_sha256': sha(guard), 'profile_sha256': sha(profile),
        'native_gpu_tests_executed': False,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()

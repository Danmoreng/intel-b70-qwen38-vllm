"""CPU-only candidate checks; no model weights or GPU devices are required."""
import ctypes
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import runpy
import subprocess
import sys

import torch
import vllm
import yaml

ROOT = Path('/opt/b70-direct')
ENGINE = Path('/opt/exl3xpu')
lock = json.loads((ROOT / 'build-lock.json').read_text())
assert vllm.__version__ == lock['latest_stable']['version']
assert bool(torch._C._GLIBCXX_USE_CXX11_ABI)
assert torch.version.git_version == lock['torch_git_revision']
assert not hasattr(torch, '_b70_graph_recording_patch')
assert hashlib.sha256(Path(torch.__file__).read_bytes()).hexdigest() == lock['pristine_torch_init_sha256']
pti_receipt = json.loads((ROOT / 'pti-patch.json').read_text())
for path, expected in pti_receipt['library_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
pti = ctypes.CDLL('/opt/venv/lib/libpti_view.so')
pti.ptiVersionString.restype = ctypes.c_char_p
assert pti.ptiVersionString().decode() == pti_receipt['version']
for name, version in {**lock['core_packages'], **lock['python_packages']}.items():
    assert metadata.version(name) == version, (name, metadata.version(name), version)
for name, version in lock['dpkg_packages'].items():
    actual = subprocess.check_output(['dpkg-query', '-W', '-f=${Version}', name], text=True)
    assert actual == version, (name, actual, version)
# First three fields of oneDNN's public dnnl_version_t are semantic version ints.
class DnnlVersion(ctypes.Structure):
    _fields_ = [('major', ctypes.c_int), ('minor', ctypes.c_int), ('patch', ctypes.c_int)]

onednn = ctypes.CDLL('/opt/b70/lib/libdnnl.so')
onednn.dnnl_version.restype = ctypes.POINTER(DnnlVersion)
version = onednn.dnnl_version().contents
actual = '.'.join(str(getattr(version, key)) for key in ('major', 'minor', 'patch'))
assert actual == lock['sources']['onednn']['runtime_version'], ('oneDNN', actual)
runpy.run_path(str(ENGINE / 'scripts/verify_native_artifact.py'))
from exl3xpu import target_matrix_guard, target_runtime, attention_dispatch
target_matrix_guard.check()
root = Path(vllm.__file__).parent
for guards in (target_runtime.SOURCE_HASHES, attention_dispatch.SOURCE_HASHES):
    for name, expected in guards.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
profile = yaml.safe_load((ENGINE / 'models/qwen3.8-27b-exl3-4.00bpw/migration-target-optimized.yaml').read_text())
assert 'mode' not in profile['vllm']['compilation_config'], 'Candidate must test standard compilation after the M04 fix'
assert profile['vllm']['compilation_config']['cudagraph_mode'] == 'FULL_DECODE_ONLY', 'Native recording must remain enabled'
m04 = ENGINE / 'm04/m04.so'
assert hashlib.sha256(m04.read_bytes()).hexdigest() == profile['env']['EXL3_M04_LIBRARY_SHA256']
torch.ops.load_library(str(m04))
assert torch._C._dispatch_has_kernel_for_dispatch_key('b70_exl3_attention::shared_kv_verify_out', 'XPU')
assert any(e.name == 'exl3xpu' for e in metadata.entry_points(group='vllm.general_plugins'))
for test in ('test_target_gdn.py', 'test_attention_metadata.py', 'test_m04_output_copy.py', 'test_checkpoint_inventory.py'):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ENGINE / 'tests'), '-p', test], check=True)
print(json.dumps({'status': 'CPU_IMPORT_AND_SOURCE_CHECKS_PASS_GPU_UNTESTED',
                  'vllm': vllm.__version__, 'torch': torch.__version__,
                  'model_loaded': False, 'gpu_tests_executed': False}))

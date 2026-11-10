"""Reject stale sources, wrong Torch ABI or wrong libraries in a candidate image."""
import hashlib
import json
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
library = root/'exl3xpu/_C.so'
manifest = json.loads(library.with_suffix('.so.manifest.json').read_text())
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
assert sha(library) == manifest['library_sha256'], 'Native library differs from manifest'
assert torch.__version__ == manifest['torch_version'], 'Wrong Torch version for EXL3 native library'
assert bool(torch._C._GLIBCXX_USE_CXX11_ABI) == manifest['torch_cxx11_abi'], 'Wrong C++ ABI'
# Compiled MB4/K4 sources are separate from the frozen parent runtime guards.
source_root = root/'native-matrix-source'
for path, expected in manifest['source_sha256'].items():
    assert sha(source_root/path) == expected, f'Stale compiled matrix source: {path}'
expected_dnnl = set(manifest['oneDNN_library_sha256'].values())
assert expected_dnnl == {sha('/opt/b70/libdnnl.so.3')}, 'Different oneDNN runtime from native build'
torch.ops.load_library(str(library))
for name in manifest['registered_ops']:
    assert hasattr(torch.ops.exl3xpu_C, name), name
for name in manifest['xpu_kernel_ops']:
    assert torch._C._dispatch_has_kernel_for_dispatch_key('exl3xpu_C::'+name, 'XPU'), name
print(json.dumps({'native_artifact':'PASS', 'torch':torch.__version__, 'sha256':sha(library), 'ops':len(manifest['registered_ops'])}))

"""Record the separately compiled M04 source/ABI identity without GPU access."""
import hashlib
import json
from pathlib import Path
import subprocess
import torch

root = Path('/native')
library = root / 'm04.so'
torch.ops.load_library(str(library))
assert torch._C._dispatch_has_kernel_for_dispatch_key('b70_exl3_attention::shared_kv_verify_out', 'XPU')
ldd = subprocess.check_output(['ldd', str(library)], text=True)
assert 'not found' not in ldd
sources = Path('/m04-source')
data = {
    'status': 'COMPILE_AND_CPU_IMPORT_PASS_XPU_UNTESTED',
    'torch': torch.__version__, 'cxx11_abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
    'source_sha256': {str(p.relative_to(sources)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(sources.rglob('*')) if p.is_file()},
    'ldd': ldd, 'compiler_version': (root / 'compiler-version.txt').read_text(),
    'compiler_argv': (root / 'm04-compiler-argv.txt').read_text().splitlines(),
}
(root / 'm04.manifest.json').write_text(json.dumps(data, indent=2) + '\n')

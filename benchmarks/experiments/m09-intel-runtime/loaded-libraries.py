"""Run inside the warm engine to record loaded driver/compiler provenance."""
import hashlib
import json
from pathlib import Path
import subprocess

libraries = {}
for maps in Path('/proc').glob('[0-9]*/maps'):
    try:
        for row in maps.read_text().splitlines():
            path = row.split()[-1]
            if path.startswith('/') and any(name in path for name in ('libze_intel_gpu', 'libigc.', 'libigdfcl.', 'libigdgmm')):
                libraries[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (PermissionError, FileNotFoundError, ProcessLookupError):
        continue
if not any('libze_intel_gpu' in path for path in libraries):
    raise RuntimeError('No loaded Intel Level Zero GPU library found')
print(json.dumps({'loaded_libraries': libraries, 'packages': subprocess.check_output([
    'dpkg-query', '-W', 'intel-igc-core-2', 'intel-igc-opencl-2', 'intel-ocloc',
    'intel-opencl-icd', 'libze-intel-gpu1', 'libigdgmm12'], text=True)}, indent=2))

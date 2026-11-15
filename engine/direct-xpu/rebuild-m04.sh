#!/usr/bin/env bash
# Rebuild only M04 after the pinned native/oneDNN/PTI cache layers.
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u
python - <<'PY'
from pathlib import Path
import hashlib
p = Path('/m04-kernels/csrc/xpu/attn/xe_2/collective/chunk_prefill_epilogue.hpp')
assert hashlib.sha256(p.read_bytes()).hexdigest() == '00a3019f4bf01917727270c22548961123e89b6721d21190a6a7da27b6b2e925'
PY
patch --batch --fuzz=0 -d /m04-kernels -p1 < /build-tools/m04-empty-split.patch
cp /build-tools/m04-empty-split.patch /m04-source/patches/empty-split.patch
python - <<'PY'
from pathlib import Path
import subprocess
args = Path('/native/m04-compiler-argv.txt').read_text().splitlines()
assert args[0] == 'icpx' and args[-1] == '/native/m04.so'
subprocess.run(args, check=True)
PY
python /build-tools/write-m04-receipt.py

"""Exclusive read-only production-binary cost screen, with service recovery."""
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from common import Session, BASE, OPERATOR, recover, save

run = Path(sys.argv[1]).resolve()
if len(sys.argv) > 2 and sys.argv[2] == 'recover':
    recover(run)
    raise SystemExit()
signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
with Session(run):
    save(run / 'manifest.json', {'image': BASE, 'power_w': 180,
        'scope': 'M04 output-layout removable-region estimate; no engine mutation',
        'source_sha256': hashlib.sha256((ROOT / 'layout-cost.py').read_bytes()).hexdigest(),
        'split_policy': {'2': 32, '3': 8, '4': 16, '5': 16}})
    cmd = ['docker', 'run', '--rm', '--name', OPERATOR, '--network=none',
           '--device', '/dev/dri', '--group-add', str(Path('/dev/dri/renderD128').stat().st_gid),
           '-e', 'ZE_AFFINITY_MASK=0', '-e', 'ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE',
           '-e', 'VLLM_TARGET_DEVICE=xpu', '-v', f'{ROOT}:/src:ro',
           '-v', f'{run}:/work', '--entrypoint', 'python', BASE, '/src/layout-cost.py']
    save(run / 'command.json', cmd)
    with (run / 'probe.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)

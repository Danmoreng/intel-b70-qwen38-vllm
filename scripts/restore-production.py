#!/usr/bin/env python3
"""Start the declared EXL3 service only after current-release preflight."""

import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request


from release_integrity import CONTAINER, REPO, SERVICE, load_release


def preflight(repo=REPO):
    release = load_release(repo)
    working = subprocess.check_output(
        ['systemctl', '--user', 'show', SERVICE, '--property=WorkingDirectory', '--value'],
        text=True).strip()
    if Path(working).resolve() != Path(repo).resolve():
        raise RuntimeError('Service WorkingDirectory differs from the reviewed release checkout: ' + working)
    start = subprocess.check_output(
        ['systemctl', '--user', 'show', SERVICE, '--property=ExecStart', '--value'], text=True).strip()
    command = re.search(r'path=(.*?) ; argv\[\]=(.*?) ; ignore_errors=', start)
    launcher = str(Path(repo).resolve() / 'scripts/run-server.sh')
    if command is None or command.group(1) != launcher or command.group(2) != launcher:
        raise RuntimeError('Service ExecStart differs from the qualified EXL3 launcher: ' + start)
    # Positional shell arguments keep paths literal. Validate the same .env
    # and strict launcher used by the service before starting anything.
    subprocess.run(['bash', '-c',
        'if [[ -f "$1" ]]; then set -a; source "$1"; set +a; fi; exec python3 "$2" --check-only',
        'exl3-release-preflight', str(Path(repo) / '.env'),
        str(Path(repo) / 'scripts/run-server-exl3.py')], check=True, timeout=720)
    return release


def main() -> int:
    release = preflight()
    subprocess.run(["systemctl", "--user", "start", SERVICE], check=True, timeout=720)
    deadline = time.monotonic() + 720
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=3) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(2)
    else:
        raise TimeoutError("production health endpoint did not recover")

    inspect = json.loads(
        subprocess.check_output(["docker", "inspect", CONTAINER], text=True)
    )[0]
    if (inspect['Image'] != release['image_id'] or
            inspect['Config']['Labels']['org.local.b70.policy.sha256'] != release['policy_sha256']):
        raise RuntimeError('Started container differs from the current qualified release')
    result = {
        "health": 200,
        "image_id": inspect["Image"],
        "policy_sha256": release['policy_sha256'],
        "service": SERVICE,
        "restored_at_unix": time.time(),
    }
    run_dir = os.getenv("B70_RUN_DIR")
    if run_dir:
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "production-restored.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

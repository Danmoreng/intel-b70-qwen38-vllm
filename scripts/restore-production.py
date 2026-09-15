#!/usr/bin/env python3
"""Idempotently stop diagnostic engines and restore the frozen production service."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


EXPECTED_IMAGE_ID = "sha256:aee9857bef1f37c8f0ee136d9f89d7166201212175a8b171d958627706cf1c0b"
DIAGNOSTIC_CONTAINERS = (
    "b70-m01-long-context",
    "b70-m01-decode-trace",
    "b70-m06-screen",
    "b70-m06-confirm",
    "b70-m06-prefix-state",
    "b70-direct-output-probe",
    "b70-direct-serving-arm",
    "b70-direct-serving-196k",
    "b70-gptq-small-m-probe",
    "b70-m04-serving-arm",
    "b70-m04-control-196k",
    "b70-m04-serving-196k",
)


def main() -> int:
    for name in DIAGNOSTIC_CONTAINERS:
        subprocess.run(
            ["docker", "stop", "-t", "30", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=45,
        )
    subprocess.run(["systemctl", "--user", "start", "qwen38.service"], check=True, timeout=720)
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
        subprocess.check_output(["docker", "inspect", "qwen38-vllm-production"], text=True)
    )[0]
    if inspect["Image"] != EXPECTED_IMAGE_ID:
        raise RuntimeError(f"restored unexpected image {inspect['Image']}")
    caps = list(Path("/sys/bus/pci/devices/0000:03:00.0/hwmon").glob("*/power1_cap"))
    cap = int(caps[0].read_text()) if len(caps) == 1 else None
    if cap != 180_000_000:
        raise RuntimeError(f"restored with unexpected power cap {cap}")
    result = {
        "health": 200,
        "image_id": inspect["Image"],
        "power_cap_uw": cap,
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

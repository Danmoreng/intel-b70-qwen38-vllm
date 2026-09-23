#!/usr/bin/env python3
"""Activate the qualified XPU-wheel image, rolling back on failed startup."""

import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location("m19_diag_promote", REPO / "scripts/run-diagnostics.py")
diag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diag
spec.loader.exec_module(diag)
ROOT = Path("/home/sebastian/LocalLLM/Local-AI-B70/qwen38")
UNIT = Path("/home/sebastian/.config/systemd/user/qwen38.service")
SOURCE_UNIT = ROOT / "production/qwen38.service"
LAUNCHER = ROOT / "run-vllm-production.sh"
OLD_TAG = "local/qwen38-b70-vllm:vllm-0.30.0-20260923"
OLD_ID = "sha256:cd6562f03c8328fe60ca69269d0e4175a284859e56525950fbfc021be19d73f3"
NEW_TAG = "local/qwen38-b70-vllm:vllm-0.30.0-xpu-kernels-0.1.15.4"
NEW_ID = "sha256:648132c9b9da4bb244d7304b956c1a9bb825be640a92755a5ffe32f2bbd679b4"


def replace_once(path, old, new):
    content = path.read_text()
    if content.count(old) != 1:
        raise RuntimeError(f"expected exactly one {old!r} in {path}")
    path.write_text(content.replace(old, new))


def check_running(expected):
    if subprocess.check_output(
        ["systemctl", "--user", "is-active", "qwen38.service"], text=True
    ).strip() != "active":
        raise RuntimeError("service is not active")
    diag.api("/health", base=diag.PRODUCTION_URL)
    actual = diag.production_inspect()
    if actual["Image"] != expected:
        raise RuntimeError(f"wrong running image: {actual['Image']}")
    if diag.api("/v1/models", base=diag.PRODUCTION_URL)["data"][0]["max_model_len"] != 200704:
        raise RuntimeError("context limit changed")
    if int(diag.POWER_CAP.read_text()) != 180_000_000:
        raise RuntimeError("power cap changed")
    return actual


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SNAPSHOT_DIR")
    snapshot = Path(sys.argv[1]).resolve()
    before = json.loads((snapshot / "snapshot.json").read_text())
    if before["previous_image"] != OLD_ID:
        raise RuntimeError("wrong rollback snapshot")
    lock = Path(
        "/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock"
    ).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_running(OLD_ID)
    diag.ensure_idle(diag.PRODUCTION_URL)
    if subprocess.check_output(
        ["docker", "image", "inspect", NEW_TAG, "--format", "{{.Id}}"], text=True
    ).strip() != NEW_ID:
        raise RuntimeError("candidate image changed")
    if SOURCE_UNIT.read_bytes() != UNIT.read_bytes():
        raise RuntimeError("installed and source units differ before promotion")
    (ROOT / "production/cache/runtime2635-xpu0115/vllm").mkdir(parents=True, exist_ok=True)
    (ROOT / "production/cache/runtime2635-xpu0115/triton").mkdir(parents=True, exist_ok=True)
    try:
        for path in (UNIT, SOURCE_UNIT):
            replace_once(path, f"Environment=VLLM_IMAGE={OLD_TAG}",
                         f"Environment=VLLM_IMAGE={NEW_TAG}")
        replace_once(LAUNCHER, f"${{VLLM_IMAGE:-{OLD_TAG}}}",
                     f"${{VLLM_IMAGE:-{NEW_TAG}}}")
        replace_once(LAUNCHER, "production/cache/runtime2635/vllm",
                     "production/cache/runtime2635-xpu0115/vllm")
        replace_once(LAUNCHER, "production/cache/runtime2635/triton",
                     "production/cache/runtime2635-xpu0115/triton")
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, timeout=30)
        subprocess.run(["systemctl", "--user", "restart", "qwen38.service"],
                       check=True, timeout=900)
        actual = check_running(NEW_ID)
        versions = subprocess.check_output([
            "docker", "exec", "qwen38-vllm-production", "python", "-c",
            "import importlib.metadata as m; print(m.version('vllm')); print(m.version('torch')); print(m.version('vllm-xpu-kernels'))",
        ], text=True, timeout=30).splitlines()
        if versions != ["0.30.0+xpu", "2.13.0+xpu", "0.1.15.4"]:
            raise RuntimeError(f"unexpected versions: {versions}")
        result = {
            "status": "active", "image": actual["Image"],
            "tag": NEW_TAG, "vllm": versions[0], "torch": versions[1],
            "vllm_xpu_kernels": versions[2], "max_model_len": 200704,
            "power_cap_uw": 180000000,
        }
        (snapshot / "promotion.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    except BaseException:
        shutil.copy2(snapshot / "installed.service", UNIT)
        shutil.copy2(snapshot / "qwen38.service", SOURCE_UNIT)
        shutil.copy2(snapshot / "run-vllm-production.sh", LAUNCHER)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, timeout=30)
        subprocess.run(["systemctl", "--user", "restart", "qwen38.service"],
                       check=True, timeout=900)
        restored = check_running(OLD_ID)
        (snapshot / "rollback.json").write_text(json.dumps({
            "status": "restored", "image": restored["Image"],
        }, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()

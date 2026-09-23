#!/usr/bin/env python3
"""Check vision/tools, prefix state, and one 192K request before promotion."""

import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location("m19_diag_qualify", REPO / "scripts/run-diagnostics.py")
diag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diag
spec.loader.exec_module(diag)
CONTROL_ID = "sha256:cd6562f03c8328fe60ca69269d0e4175a284859e56525950fbfc021be19d73f3"
CANDIDATE = "local/qwen38-b70-vllm:vllm-0.30.0-xpu-kernels-0.1.15.4"
CANDIDATE_ID = "sha256:648132c9b9da4bb244d7304b956c1a9bb825be640a92755a5ffe32f2bbd679b4"
NAME = "b70-m19-xpu-qualification"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RUN_DIR")
    run = Path(sys.argv[1]).resolve()
    run.mkdir(parents=True, exist_ok=False)
    lock = Path(
        "/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock"
    ).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    production = diag.production_inspect()
    if production["Image"] != CONTROL_ID:
        raise RuntimeError("production image changed")
    candidate_id = subprocess.check_output(
        ["docker", "image", "inspect", CANDIDATE, "--format", "{{.Id}}"], text=True
    ).strip()
    if candidate_id != CANDIDATE_ID:
        raise RuntimeError("candidate image changed")
    diag.ensure_idle(diag.PRODUCTION_URL)
    if int(diag.POWER_CAP.read_text()) != 180_000_000:
        raise RuntimeError("power cap changed")
    command = diag.engine_command(
        production, name=NAME, image=CANDIDATE, evidence=run,
        arguments=production["Args"].copy(),
    )
    for index, value in enumerate(command):
        if value.endswith(":/root/.cache/vllm") or value.endswith(":/root/.triton/cache"):
            destination = value.split(":", 1)[1]
            kind = "vllm" if destination.endswith("/vllm") else "triton"
            cache = run / "compiler-cache" / kind
            cache.mkdir(parents=True, exist_ok=True)
            command[index] = f"{cache}:{destination}"
    (run / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    process = None
    server_log = None
    stopped_production = False
    try:
        stopped_production = True
        subprocess.run(["systemctl", "--user", "stop", "qwen38.service"],
                       check=True, timeout=120)
        server_log = (run / "server.log").open("w")
        process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
        diag.wait_ready(process, timeout=900)
        print("CANDIDATE READY", flush=True)
        checks = [
            (
                "vision-tools",
                [sys.executable,
                 "/home/sebastian/LocalLLM/Local-AI-B70/qwen38/production/check-vision-tools.py",
                 "--root", diag.EXPERIMENT_URL,
                 "--output", str(run / "vision-tools.json")],
                300,
            ),
            (
                "prefix-cache",
                [sys.executable,
                 "/home/sebastian/LocalLLM/Local-AI-B70/qwen38/production/check-prefix-cache.py",
                 "--root", diag.EXPERIMENT_URL, "--records", "800",
                 "--output", str(run / "prefix-cache.json")],
                300,
            ),
            (
                "long-context",
                [sys.executable, str(REPO / "scripts/current-profile-benchmark.py"),
                 "--execute", "--base", diag.EXPERIMENT_URL,
                 "--container", NAME,
                 "--scenarios", str(REPO / "benchmarks/experiments/m15-q128-long-context/serving-scenario.json"),
                 "--output-root", str(run / "long-context")],
                900,
            ),
        ]
        for label, command, timeout in checks:
            print("START", label, flush=True)
            with (run / f"{label}.log").open("w") as output:
                subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                               check=True, timeout=timeout)
            print("PASS", label, flush=True)
        log = (run / "server.log").read_text()
        if "B70_Q128_DISPATCH" not in log or "B70_M04_SHARED_KV_DISPATCH" not in log:
            raise RuntimeError("custom attention dispatch missing")
        (run / "qualification.json").write_text(json.dumps({
            "candidate_image": CANDIDATE_ID,
            "vision_tools": "PASS", "prefix_cache": "PASS",
            "long_context_196608_input": "PASS",
            "q128_m04_dispatch": "PASS",
        }, indent=2) + "\n")
    finally:
        try:
            subprocess.run(["docker", "stop", "-t", "30", NAME],
                           capture_output=True, timeout=50)
            if process is not None:
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        finally:
            if server_log is not None:
                server_log.close()
            if stopped_production:
                subprocess.run(["systemctl", "--user", "start", "qwen38.service"],
                               check=True, timeout=900)
                diag.ensure_idle(diag.PRODUCTION_URL)
                actual = diag.production_inspect()
                if actual["Image"] != CONTROL_ID:
                    raise RuntimeError("production restored with wrong image")
                (run / "production-restored.json").write_text(json.dumps({
                    "image": actual["Image"],
                    "power_cap_uw": int(diag.POWER_CAP.read_text()),
                    "active": True,
                }, indent=2) + "\n")


if __name__ == "__main__":
    main()

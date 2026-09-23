#!/usr/bin/env python3
"""Short 0.1.14.1/0.1.15.4 XPU-kernel serving A/B; restore production."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


diag = load("m19_diag", REPO / "scripts/run-diagnostics.py")
measure = load("m19_wiki", REPO / "benchmarks/experiments/m13-wikipedia-mtp/measure.py")
CONTROL = "local/qwen38-b70-vllm:vllm-0.30.0-20260923"
CONTROL_ID = "sha256:cd6562f03c8328fe60ca69269d0e4175a284859e56525950fbfc021be19d73f3"
CANDIDATE = "local/qwen38-b70-vllm:vllm-0.30.0-xpu-kernels-0.1.15.4"
CANDIDATE_ID = "sha256:648132c9b9da4bb244d7304b956c1a9bb825be640a92755a5ffe32f2bbd679b4"
NAME = "b70-m19-xpu-kernels-arm"
SAMPLING = {"temperature": 1, "top_p": 0.95, "top_k": 20, "max_tokens": 512}
ORDER = ("candidate", "control", "control", "candidate")


def image_id(tag):
    return subprocess.check_output(
        ["docker", "image", "inspect", tag, "--format", "{{.Id}}"], text=True
    ).strip()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if image_id(CONTROL) != CONTROL_ID or image_id(CANDIDATE) != CANDIDATE_ID:
        parser.error("image digest changed")
    articles = json.loads(args.prompts.read_text())
    if len(articles) != 4 or any(
        hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]
        for row in articles
    ):
        parser.error("frozen Wikipedia prompts changed")
    run = args.run_dir.resolve()
    plan = {
        "control_image": CONTROL_ID, "candidate_image": CANDIDATE_ID,
        "vllm": "0.30.0+xpu", "torch": "2.13.0+xpu",
        "kernel_versions": {"control": "0.1.14.1", "candidate": "0.1.15.4"},
        "sampling": SAMPLING, "mtp": 4, "order": ORDER,
        "waves_per_arm": ["C4 warmup", "C1 × 2", "C4 × 2"],
        "max_run_minutes_before_recovery": 27,
        "articles": [{key: row[key] for key in (
            "title", "revision_id", "prompt_sha256", "prompt_tokens")}
            for row in articles],
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    run.mkdir(parents=True, exist_ok=False)
    save(run / "plan.json", plan)
    lock = Path(
        "/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock"
    ).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    production = diag.production_inspect()
    if production["Image"] != CONTROL_ID:
        raise RuntimeError("unexpected production image")
    diag.ensure_idle(diag.PRODUCTION_URL)
    if int(diag.POWER_CAP.read_text()) != 180_000_000:
        raise RuntimeError("unexpected power cap")
    measure.SAMPLING = SAMPLING
    deadline = time.monotonic() + 27 * 60
    process = None
    server_log = None
    stopped_production = False
    try:
        stopped_production = True
        subprocess.run(["systemctl", "--user", "stop", "qwen38.service"],
                       check=True, timeout=120)
        for number, arm in enumerate(ORDER, 1):
            if time.monotonic() > deadline - 300:
                raise TimeoutError("screen budget exhausted before next arm")
            image = CANDIDATE if arm == "candidate" else CONTROL
            arm_dir = run / f"{number:02d}-{arm}"
            arm_dir.mkdir()
            command = diag.engine_command(
                production, name=NAME, image=image, evidence=arm_dir,
                arguments=production["Args"].copy(),
            )
            for index, value in enumerate(command):
                if value.endswith(":/root/.cache/vllm") or value.endswith(":/root/.triton/cache"):
                    destination = value.split(":", 1)[1]
                    kind = "vllm" if destination.endswith("/vllm") else "triton"
                    cache = run / "compiler-cache" / arm / kind
                    cache.mkdir(parents=True, exist_ok=True)
                    command[index] = f"{cache}:{destination}"
            at = command.index(image)
            command[at:at] = ["-e", "VLLM_SERVER_DEV_MODE=1", "-e", "PYTHONUNBUFFERED=1"]
            save(arm_dir / "command.json", command)
            server_log = (arm_dir / "server.log").open("w")
            process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
            print(f"START {number:02d} {arm}", flush=True)
            diag.wait_ready(process, timeout=min(900, max(60, int(deadline-time.monotonic()))))
            version_text = subprocess.check_output([
                "docker", "exec", NAME, "python", "-c",
                "import importlib.metadata as m; print(m.version('vllm')); print(m.version('vllm-xpu-kernels')); print(m.version('torch'))",
            ], text=True, timeout=30)
            (arm_dir / "versions.txt").write_text(version_text)
            measured = arm_dir / "benchmark"
            measured.mkdir()
            measure.wave(diag.EXPERIMENT_URL, measured, articles, 4, 0, warmup=True)
            rows = [
                measure.wave(diag.EXPERIMENT_URL, measured, articles, concurrency, repeat)
                for concurrency in (1, 4) for repeat in (0, 1)
            ]
            save(arm_dir / "results.json", {"rows": rows})
            subprocess.run(["docker", "stop", "-t", "30", NAME],
                           capture_output=True, timeout=50)
            process.wait(timeout=60)
            process = None
            server_log.close()
            server_log = None
            log = (arm_dir / "server.log").read_text()
            if "B70_Q128_DISPATCH" not in log or "B70_M04_SHARED_KV_DISPATCH" not in log:
                raise RuntimeError(f"custom Q128/M04 dispatch missing in {arm}")
            print(f"DONE {number:02d} {arm}", flush=True)
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
                restored = diag.production_inspect()
                if restored["Image"] != CONTROL_ID:
                    raise RuntimeError("production restored with wrong image")
                save(run / "production-restored.json", {
                    "image": restored["Image"], "active": True,
                    "power_cap_uw": int(diag.POWER_CAP.read_text()),
                })


if __name__ == "__main__":
    main()

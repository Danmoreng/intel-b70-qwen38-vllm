#!/usr/bin/env python3
"""Run the frozen C3 trace after C2 passes, on three fresh candidate workers."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "benchmark-results/production-release-v1/integrated-serving-v1"
C2 = REPO / "benchmark-results/production-release-v1/heldout-quality-v2"
LOCK = Path("/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock")
SERVICE = "b70-candidate-qualification-v2.service"
CONTAINER = "b70-qwen38-vllm"
IMAGE = "sha256:a42cda993bf6492acc39d23e9382e27a17efca4657bea07d80d0c28228a1623a"
BASE = "http://127.0.0.1:8081"


def save(name: str, value):
    target = ROOT / name
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(target)


def status(state: str, detail: str):
    save("status.json", {"state": state, "detail": detail,
                         "updated_unix": time.time()})
    print(f"{state}: {detail}", flush=True)


def check_c2():
    start = time.monotonic()
    while time.monotonic() - start < 14400:
        if (C2 / "status.json").exists():
            state = json.loads((C2 / "status.json").read_text())
            if state["state"] == "failed":
                raise RuntimeError(f"C2 failed: {state['detail']}")
            if state["state"] == "complete":
                paired = json.loads((C2 / "paired-summary.json").read_text())
                if not paired["practical_quality_gate_passed"]:
                    raise RuntimeError(f"C2 quality gate failed: {paired}")
                return paired
        time.sleep(20)
    raise TimeoutError("C2 did not finish within four hours")


def inspect():
    result = subprocess.run(["docker", "inspect", CONTAINER],
                            capture_output=True, text=True, timeout=20)
    return json.loads(result.stdout)[0] if result.returncode == 0 else None


def wait_ready(previous_id: str | None):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        item = inspect()
        if item:
            if item["Image"] != IMAGE:
                raise RuntimeError(f"unexpected image: {item['Image']}")
            if item["State"]["Status"] == "exited":
                raise RuntimeError("candidate worker exited")
            if item["Id"] != previous_id:
                try:
                    with urllib.request.urlopen(BASE + "/health", timeout=2) as response:
                        if response.status == 200:
                            return item["Id"]
                except OSError:
                    pass
        time.sleep(3)
    raise TimeoutError("fresh candidate worker not healthy")


def assert_idle():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()
    for metric in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        total = sum(float(line.split()[-1]) for line in lines
                    if line.startswith(metric + "{") or line.startswith(metric + " "))
        if total:
            raise RuntimeError(f"candidate is busy: {metric}={total}")


def main():
    if ROOT.exists():
        raise RuntimeError("integrated serving result directory already exists")
    ROOT.mkdir(parents=True)
    try:
        status("waiting", "C2 paired quality gate")
        paired = check_c2()
        save("c2-paired-summary.json", paired)
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        with LOCK.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if subprocess.run(["systemctl", "--user", "is-active", "qwen38.service"],
                              capture_output=True).returncode == 0:
                raise RuntimeError("permanent production service unexpectedly active")
            previous = wait_ready(None)
            assert_idle()
            worker_ids = []
            for index in (1, 2, 3):
                status("running", f"restart and worker {index}/3")
                subprocess.run(["systemctl", "--user", "restart", SERVICE],
                               check=True, timeout=180)
                current = wait_ready(previous)
                assert_idle()
                worker_ids.append(current)
                previous = current
                worker_dir = ROOT / f"worker-{index}"
                with (ROOT / f"worker-{index}.log").open("w") as log:
                    subprocess.run([sys.executable,
                                    str(REPO / "scripts/run-integrated-serving-worker.py"),
                                    "--worker-index", str(index),
                                    "--output", str(worker_dir)],
                                   check=True, cwd=REPO, stdout=log,
                                   stderr=subprocess.STDOUT)
            rows = [json.loads(path.read_text()) for path in ROOT.glob("worker-*/*.json")
                    if path.name != "identity.json"]
            failures = {row["id"]: row["failures"] for row in rows if row["failures"]}
            counts = {}
            for row in rows:
                counts[row["kind"]] = counts.get(row["kind"], 0) + len(row["requests"])
            result = {"groups": len(rows), "requests": sum(counts.values()),
                      "counts": counts, "distinct_worker_ids": len(set(worker_ids)),
                      "worker_ids": worker_ids, "failures": failures}
            save("trace-summary.json", result)
            if len(rows) != 14 or result["requests"] != 40 or len(set(worker_ids)) != 3 or failures:
                raise RuntimeError(f"integrated serving trace failed: {result}")
            status("complete", "40-request trace passed on three fresh workers")
    except Exception as error:
        status("failed", f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()

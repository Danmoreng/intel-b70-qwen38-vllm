#!/usr/bin/env python3
"""Run the isolated M04 shared-KV verification-attention gate."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


REPO = Path(__file__).resolve().parents[1]
IMAGE = "local/qwen38-b70-vllm:q128-196k-20260914"
IMAGE_ID = "sha256:3f20b0cf493fe0904a7efd0ca310067790e901bc5d60203340c411e57c25010a"
CONTAINER = "b70-shared-kv-verification-probe"


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    state = {"status": "preflight", "started_at_unix": time.time(),
             "run_dir": str(run)}
    save(run / "state.json", state)

    lock_path = Path("/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock")
    lock = lock_path.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,
                  lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    actual_id = subprocess.check_output(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
        text=True).strip()
    if actual_id != IMAGE_ID:
        raise RuntimeError(f"runtime image changed: {actual_id}")
    cap = next(Path("/sys/bus/pci/devices/0000:03:00.0/hwmon").glob("*/power1_cap"))
    if int(cap.read_text()) != 180000000:
        raise RuntimeError("power cap is not 180 W")
    raw = urllib.request.urlopen("http://127.0.0.1:8081/metrics",
                                 timeout=10).read().decode()
    for metric in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        values = [float(line.split()[-1]) for line in raw.splitlines()
                  if line.startswith(metric + "{") or
                  line.startswith(metric + " ")]
        if not values or sum(values):
            raise RuntimeError(f"production busy: {metric}={values}")

    (run / "b70_ops.so").write_bytes(
        (REPO / "b70_ops/build/b70_ops.so").read_bytes())
    command = [
        "docker", "run", "--rm", "--name", CONTAINER, "--network", "none",
        "--device", "/dev/dri", "--group-add",
        str(Path("/dev/dri/renderD128").stat().st_gid),
        "-e", "ZE_AFFINITY_MASK=0", "-e",
        "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
        "-v", f"{run}:/work",
        "-v", f"{REPO / 'b70_ops/probe_shared_kv_split.py'}:/probe.py:ro",
        "--entrypoint", "python", IMAGE, "/probe.py",
    ]
    save(run / "command.json", command)
    try:
        state["status"] = "stopping-production"
        save(run / "state.json", state)
        subprocess.run(["systemctl", "--user", "stop", "qwen38.service"],
                       check=True, timeout=120)
        state["status"] = "operator-screen"
        save(run / "state.json", state)
        with (run / "probe.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=3600)
        checks = json.loads((run / "split-correctness.json").read_text())
        results = json.loads((run / "split-results.json").read_text())
        q5 = [row for row in results["rows"] if row["q_len"] == 5]
        assessment = {
            "all_correct": len(checks) == 90 and all(
                row["challenger_vs_deployed"]["finite"] and
                row["challenger_vs_deployed"]["allclose"]
                for row in checks),
            "q5_speedups_pct": [row["best_speedup_pct"] for row in q5],
        }
        assessment["pass"] = (
            assessment["all_correct"] and len(q5) == len(CONTEXTS) and
            all(value >= 3 for value in assessment["q5_speedups_pct"]))
        state["assessment"] = assessment
        state["status"] = ("qualified-for-serving" if assessment["pass"]
                           else "rejected")
        state["finished_at_unix"] = time.time()
        save(run / "state.json", state)
        return 0
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        state["finished_at_unix"] = time.time()
        save(run / "state.json", state)
        raise
    finally:
        subprocess.run(["docker", "stop", "-t", "30", CONTAINER],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=45)
        subprocess.run(["python3", str(REPO / "scripts/restore-production.py")],
                       check=True, timeout=800,
                       env={**os.environ, "B70_RUN_DIR": str(run)})


CONTEXTS = (8192, 65536, 131072, 196608)


if __name__ == "__main__":
    raise SystemExit(main())

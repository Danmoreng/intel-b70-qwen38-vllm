#!/usr/bin/env python3
"""Run the M05 offline screen with unconditional production restoration."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "benchmarks/experiments/m05-int2-draft-head"
SNAPSHOT = "a47b0c6f0d756bc394c4cc629d5b0ded1acc7001"
MODEL = (Path.home() / ".cache/huggingface/hub/models--mikeinnyc--Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
         / "snapshots" / SNAPSHOT)
IMAGE = "local/qwen38-b70-vllm:q128-m04-196k-20260915"


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


stamp = dt.datetime.now().strftime("run-%Y%m%d-%H%M%S")
run = EXP / "runs" / stamp
run.mkdir(parents=True)
captures = sorted(path for path in (EXP / "runs").glob("capture-*/draft.pt"))
hidden = captures[-1] if captures else None
(EXP / "LATEST").unlink(missing_ok=True)
(EXP / "LATEST").symlink_to(Path("runs") / stamp)
state = {"status": "starting", "run": str(run), "image": IMAGE,
         "started_at": time.time(), "hidden_states": str(hidden) if hidden else None}
write(run / "state.json", state)

returncode = None
try:
    subprocess.run(["systemctl", "--user", "stop", "qwen38.service"], check=True,
                   timeout=120)
    state["status"] = "running"
    write(run / "state.json", state)
    gid = os.stat("/dev/dri/renderD128").st_gid
    command = [
        "docker", "run", "--rm", "--device", "/dev/dri", "--group-add", str(gid),
        "-e", "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE", "-e", "ZE_AFFINITY_MASK=0",
        "-v", f"{Path.home() / '.cache/huggingface'}:/hf:ro",
        "-v", f"{EXP}:/work:ro", "-v", f"{run}:/output",
        "--entrypoint", "python", IMAGE, "/work/int2_probe.py",
        "--weight", f"/hf/hub/{MODEL.parent.parent.name}/snapshots/{SNAPSHOT}/model-00002-of-00005.safetensors",
        "--output", "/output/results.json",
    ]
    if hidden is not None:
        command += ["--hidden-states", f"/work/runs/{hidden.parent.name}/draft.pt"]
    with (run / "benchmark.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=3600)
    returncode = completed.returncode
    state["status"] = "complete" if returncode == 0 else "failed"
    state["returncode"] = returncode
except Exception as error:
    state["status"] = "failed"
    state["error"] = repr(error)
finally:
    state["finished_at"] = time.time()
    write(run / "state.json", state)
    subprocess.run(["systemctl", "--user", "start", "qwen38.service"], timeout=720)
    deadline = time.time() + 720
    health = None
    while time.time() < deadline:
        try:
            health = urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=5).status
            if health == 200:
                break
        except Exception:
            time.sleep(2)
    write(run / "restored.json", {"health": health, "time": time.time()})

raise SystemExit(0 if state["status"] == "complete" and health == 200 else 1)

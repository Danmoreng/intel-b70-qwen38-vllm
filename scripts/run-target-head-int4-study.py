#!/usr/bin/env python3
"""Run M06 offline and unconditionally restore production."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


REPO = Path(__file__).resolve().parents[1]
M05 = REPO / "benchmarks/experiments/m05-int2-draft-head"
EXP = REPO / "benchmarks/experiments/m06-target-head-int4"
IMAGE = "local/qwen38-b70-vllm:q128-m04-196k-20260915"
REV = "a47b0c6f0d756bc394c4cc629d5b0ded1acc7001"
capture = sorted((M05 / "runs").glob("capture-*/target.pt"))[-1]
stamp = dt.datetime.now().strftime("run-%Y%m%d-%H%M%S")
run = EXP / "runs" / stamp
run.mkdir(parents=True)
(EXP / "LATEST").unlink(missing_ok=True)
(EXP / "LATEST").symlink_to(Path("runs") / stamp)
state = {"status": "starting", "run": str(run), "capture": str(capture),
         "started_at": time.time()}
(run / "state.json").write_text(json.dumps(state, indent=2) + "\n")
health = None
try:
    subprocess.run(["systemctl", "--user", "stop", "qwen38.service"],
                   check=True, timeout=120)
    state["status"] = "running"
    (run / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    gid = os.stat("/dev/dri/renderD128").st_gid
    model = "models--mikeinnyc--Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
    command = ["docker", "run", "--rm", "--device", "/dev/dri",
               "--group-add", str(gid), "-e", "ZE_AFFINITY_MASK=0",
               "-v", f"{Path.home() / '.cache/huggingface'}:/hf:ro",
               "-v", f"{REPO}:/repo:ro", "-v", f"{run}:/output",
               "--entrypoint", "python", IMAGE, "/repo/benchmarks/experiments/m06-target-head-int4/target_head_probe.py",
               "--weight", f"/hf/hub/{model}/snapshots/{REV}/model-00002-of-00005.safetensors",
               "--hidden-states", f"/repo/{capture.relative_to(REPO)}",
               "--output", "/output/results.json"]
    with (run / "benchmark.log").open("w") as log:
        complete = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=3600)
    state["status"] = "complete" if complete.returncode == 0 else "failed"
    state["returncode"] = complete.returncode
except Exception as error:
    state["status"] = "failed"
    state["error"] = repr(error)
finally:
    state["finished_at"] = time.time()
    (run / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    subprocess.run(["systemctl", "--user", "start", "qwen38.service"], timeout=720)
    deadline = time.time() + 720
    while time.time() < deadline:
        try:
            health = urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=5).status
            if health == 200:
                break
        except Exception:
            time.sleep(2)
    (run / "restored.json").write_text(json.dumps({"health": health}, indent=2) + "\n")

raise SystemExit(0 if state["status"] == "complete" and health == 200 else 1)

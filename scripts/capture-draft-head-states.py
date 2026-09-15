#!/usr/bin/env python3
"""Capture real draft/target head inputs and restore production."""
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
IMAGE = "local/qwen38-b70-vllm:m05-capture"
MODEL = "mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
REV = "a47b0c6f0d756bc394c4cc629d5b0ded1acc7001"


stamp = dt.datetime.now().strftime("capture-%Y%m%d-%H%M%S")
run = EXP / "runs" / stamp
run.mkdir(parents=True)
state = {"status": "building", "run": str(run), "started_at": time.time()}
(run / "state.json").write_text(json.dumps(state, indent=2) + "\n")
health = None
container = "b70-m05-capture"
try:
    subprocess.run(["docker", "build", "-f", str(EXP / "capture/Dockerfile"),
                    "-t", IMAGE, str(REPO)], check=True, timeout=600,
                   stdout=(run / "build.log").open("w"), stderr=subprocess.STDOUT)
    subprocess.run(["systemctl", "--user", "stop", "qwen38.service"],
                   check=True, timeout=120)
    state["status"] = "loading"
    (run / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    gid = os.stat("/dev/dri/renderD128").st_gid
    command = [
        "docker", "run", "--rm", "--name", container, "--device", "/dev/dri",
        "--group-add", str(gid), "--shm-size", "8g", "-p", "127.0.0.1:18005:8000",
        "-v", f"{Path.home() / '.cache/huggingface'}:/root/.cache/huggingface",
        "-v", f"{run}:/evidence", "-e", "HF_HUB_OFFLINE=1",
        "-e", "VLLM_TARGET_DEVICE=xpu", "-e", "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
        "-e", "ZE_AFFINITY_MASK=0", "-e", "B70_DRAFT_LMHEAD_INT4=1",
        "-e", "B70_DRAFT_MTP_INT4=1", "-e", "B70_DRAFT_VOCAB_ENABLED=0",
        "-e", "B70_MTP_BF16_DRAFT=1", "-e", "VLLM_XPU_ENABLE_XPU_GRAPH=0",
        "--entrypoint", "vllm", IMAGE, "serve", MODEL, "--revision", REV,
        "--quantization", "gptq", "--dtype", "float16", "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.85", "--kv-cache-dtype", "fp8",
        "--max-num-seqs", "1", "--max-num-batched-tokens", "4096",
        "--mamba-cache-mode", "align", "--served-model-name", "Qwen3.8-27B",
        "--speculative-config", '{"method":"mtp","num_speculative_tokens":4}',
        "--enforce-eager", "--reasoning-config",
        '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}',
    ]
    log = (run / "server.log").open("w")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            if urllib.request.urlopen("http://127.0.0.1:18005/health", timeout=5).status == 200:
                break
        except Exception:
            if server.poll() is not None:
                raise RuntimeError(f"capture server exited {server.returncode}")
            time.sleep(2)
    else:
        raise TimeoutError("capture server did not become ready")
    (run / "capture").touch()
    body = {"model": "Qwen3.8-27B", "prompt": "Count upward: 1, 2, 3,",
            "max_tokens": 160, "temperature": 0, "ignore_eos": True}
    request = urllib.request.Request("http://127.0.0.1:18005/v1/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    (run / "request.json").write_text(json.dumps(result, indent=2) + "\n")
    (run / "done").touch()
    for _ in range(100):
        if (run / "draft.pt").exists() and (run / "target.pt").exists():
            break
        time.sleep(0.1)
    if not (run / "draft.pt").exists() or not (run / "target.pt").exists():
        raise RuntimeError("request ended without both activation captures")
    state["status"] = "complete"
except Exception as error:
    state["status"] = "failed"
    state["error"] = repr(error)
    raise
finally:
    subprocess.run(["docker", "stop", "-t", "30", container], timeout=60,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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

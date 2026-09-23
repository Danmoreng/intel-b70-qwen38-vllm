#!/usr/bin/env python3
"""One short vLLM 0.30 trace with XPU graph disabled; always restore production."""

import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("m18_diag", REPO / "scripts/run-diagnostics.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)
IMAGE = "sha256:cd6562f03c8328fe60ca69269d0e4175a284859e56525950fbfc021be19d73f3"
NAME = "b70-gdn-output-trace"


def main():
    run = Path(sys.argv[1]).resolve()
    run.mkdir(parents=True, exist_ok=False)
    lock = Path("/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    production = diag.production_inspect()
    if production["Image"] != IMAGE:
        raise RuntimeError(f"unexpected production image {production['Image']}")
    diag.ensure_idle(diag.PRODUCTION_URL)
    if int(diag.POWER_CAP.read_text()) != 180000000:
        raise RuntimeError("unexpected power cap")
    args = production["Args"].copy()
    profiler = {
        "profiler": "torch", "torch_profiler_dir": "/evidence/trace",
        "torch_profiler_with_stack": False,
        "torch_profiler_record_shapes": True,
        "torch_profiler_dump_cuda_time_total": False,
        "ignore_frontend": True,
    }
    args += ["--profiler-config", json.dumps(profiler)]
    command = diag.engine_command(
        production, name=NAME, image=production["Config"]["Image"],
        evidence=run, arguments=args,
    )
    for i, value in enumerate(command):
        if value.startswith("VLLM_XPU_ENABLE_XPU_GRAPH="):
            command[i] = "VLLM_XPU_ENABLE_XPU_GRAPH=0"
    (run / "manifest.json").write_text(json.dumps({
        "production_image": production["Image"],
        "production_command": production["Args"],
        "diagnostic_command": command,
        "reason_for_graph_disable": "Expose individual decode kernels to PyTorch profiler",
    }, indent=2) + "\n")
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
        prompt = "B70 GDN output-copy trace. Explain this test briefly.\n" + " x" * 4096
        start = time.monotonic()
        diag.api("/start_profile", {})
        try:
            response = diag.api("/v1/completions", {
                "model": "Qwen3.8-27B", "prompt": prompt,
                "max_tokens": 32, "temperature": 0, "seed": 38,
                "ignore_eos": True,
            }, timeout=600)
        finally:
            diag.api("/stop_profile", {}, timeout=180)
        (run / "request.json").write_text(json.dumps({
            "wall_s": time.monotonic() - start,
            "usage": response.get("usage"),
            "finish_reason": response["choices"][0].get("finish_reason"),
            "output_chars": len(response["choices"][0].get("text") or ""),
        }, indent=2) + "\n")
    finally:
        try:
            if process is not None:
                subprocess.run(["docker", "stop", "-t", "30", NAME],
                               capture_output=True, timeout=50)
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
                if actual["Image"] != IMAGE:
                    raise RuntimeError("production restored with wrong image")
                (run / "production-restored.json").write_text(json.dumps({
                    "image": actual["Image"], "active": True,
                }, indent=2) + "\n")


if __name__ == "__main__":
    main()

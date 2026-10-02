#!/usr/bin/env python3
"""Exclusive sequential GPTQ/EXL3 campaign; optional pinned production restoration."""

import argparse
import fcntl
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "web_runner", REPO / "scripts/run-web-coding-benchmark.py"
)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
SERVICE = "b70-qwen38-vllm.service"
EXL_NAME = "b70-exl3xpu-coding"
EXL_IMAGE = "sha256:cba73584f4ab0a2b37eac1356f34f16655ac5e740d845e110997b03be78279b7"
EXL_MODEL = Path(
    os.environ.get(
        "EXL3_MODEL_DIR",
        str(Path.home() / ".cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw"),
    )
)


def command(argv, **kwargs):
    return subprocess.run(
        argv, check=True, timeout=kwargs.pop("timeout", 120), **kwargs
    )


def wait_health(base, container, image):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        try:
            R.http(base, "/v1/models", timeout=3)
            x = R.identity(container)
            if x["image_id"] != image:
                raise RuntimeError("worker has unexpected image")
            return x
        except (OSError, subprocess.CalledProcessError):
            time.sleep(3)
    raise RuntimeError("worker did not become ready: " + container)


def stop_exl():
    exists = subprocess.run(
        ["docker", "inspect", EXL_NAME], capture_output=True, text=True, check=False
    )
    if exists.returncode == 0:
        if json.loads(exists.stdout)[0]["Image"] != EXL_IMAGE:
            raise RuntimeError("refuse to stop unexpected container " + EXL_NAME)
        command(["docker", "stop", EXL_NAME], timeout=90)


def restore():
    stop_exl()
    release = json.loads((REPO / "config/production_image.json").read_text())
    command(["systemctl", "--user", "start", SERVICE])
    return wait_health("http://127.0.0.1:8081", "b70-qwen38-vllm", release["image_id"])


def warmup(base, model, out):
    before = R.snapshot(base)
    text = (
        "Excluded deterministic compiler warm-up. Read the following numbered notes and respond briefly.\n"
        + "".join(
            f"Warmup note {i}: stable coding benchmark startup, no fixture content.\n"
            for i in range(420)
        )
    )
    p = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 64,
        "ignore_eos": True,
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 72999,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    a = R.stream(base, p, out / "warmup.sse.jsonl")
    _, d = R.wait_accounted(base, before, a["usage"])
    R.save(
        out / "warmup.json",
        {"response": a, "native": d, "excluded_from_scored_run": True},
    )


def start_exl():
    if not (EXL_MODEL / "model.safetensors.index.json").exists():
        raise RuntimeError("missing local EXL3 model")
    cache = Path.home() / ".cache/exl3xpu/runtime"
    for d in ["vllm", "triton", "neo_compiler_cache"]:
        (cache / d).mkdir(parents=True, exist_ok=True)
    argv = [
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        EXL_NAME,
        "--device",
        "/dev/dri",
        "-v",
        "/dev/dri/by-path:/dev/dri/by-path:ro",
        "--shm-size",
        "32g",
        "-p",
        "127.0.0.1:8082:8000",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-v",
        str(EXL_MODEL) + ":/models:ro",
        "-v",
        str(cache / "vllm") + ":/root/.cache/vllm",
        "-v",
        str(cache / "triton") + ":/root/.triton/cache",
        "-v",
        str(cache / "neo_compiler_cache") + ":/root/.cache/neo_compiler_cache",
        EXL_IMAGE,
        "models/qwen3.8-27b-exl3-4.00bpw",
        "--gpu",
        "0",
        "--port",
        "8000",
        "--model-path",
        "/models",
        "--",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--reasoning-parser",
        "qwen3",
        "--reasoning-config",
        '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}',
    ]
    command(argv)
    return argv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--engines", default="gptq,exl3")
    p.add_argument("--restore-production", action="store_true")
    p.add_argument("--restore-after-campaign", action="store_true")
    p.add_argument("--max-wall-seconds", type=int, default=7200)
    p.add_argument("--thinking-token-budget", type=int, default=4096)
    a = p.parse_args()
    if a.restore_production:
        print(json.dumps(restore()))
        return
    if not a.fixture or not a.output:
        p.error("fixture and output required")
    release = json.loads((REPO / "config/production_image.json").read_text())
    if release['image_id'] != 'sha256:ed1ebca756abb0e0832d11cd0db026dd7e86df094c6903efe7ae8afbdc290b68':
        raise RuntimeError(
            'This historical campaign labels the permanent service GPTQ. '
            'Use run-exl3-final-readme.py or the standalone coding runner for the current EXL3 release; '
            'refusing to mislabel the current service or start the old EXL3 image.'
        )
    if a.output.exists():
        raise RuntimeError("fresh output required")
    engines = a.engines.split(",")
    if not set(engines) <= {"gptq", "exl3"}:
        raise ValueError("unsupported engine")
    lock_path = REPO.parent / "Local-AI-B70/qwen38/context-benchmark/run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output.mkdir(parents=True)
        (a.output / "campaign-source.py").write_bytes(Path(__file__).read_bytes())
        release = json.loads((REPO / "config/production_image.json").read_text())

        def interrupt(*_):
            raise RuntimeError("campaign interrupted; restoring production")

        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        state = {
            "status": "running",
            "started_at": R.now(),
            "engines": engines,
            "fixture": str(a.fixture.resolve()),
            "fixture_manifest_sha256": R.sha(a.fixture / "manifest.json"),
            "production_release": release,
            "exl3_image_id": EXL_IMAGE,
            "exl3_model_revision": "113cf7ab958054860e43fb7f3063b1af19171095",
            "exl3_bits": 4.0,
            "exl3_head_bits": 6,
            "run_limits_seconds_per_engine": a.max_wall_seconds,
            "thinking_token_budget": a.thinking_token_budget,
        }
        R.save(a.output / "campaign.json", state)
        try:
            for engine in engines:
                state["phase"] = engine + "-startup"
                R.save(a.output / "campaign.json", state)
                if engine == "gptq":
                    stop_exl()
                    command(["systemctl", "--user", "restart", SERVICE])
                    base = "http://127.0.0.1:8081"
                    container = "b70-qwen38-vllm"
                    live = wait_health(base, container, release["image_id"])
                else:
                    command(["systemctl", "--user", "stop", SERVICE])
                    launch = start_exl()
                    base = "http://127.0.0.1:8082"
                    container = EXL_NAME
                    live = wait_health(base, container, EXL_IMAGE)
                    R.save(
                        a.output / "exl3-launch.json",
                        {"argv": launch, "identity": live},
                    )
                R.save(a.output / (engine + "-identity.json"), live)
                models = R.http(base, "/v1/models")
                warm_root = a.output / (engine + "-warmup")
                warm_root.mkdir()
                warmup(base, models["data"][0]["id"], warm_root)
                state["phase"] = engine + "-coding"
                R.save(a.output / "campaign.json", state)
                try:
                    command(
                        [
                            sys.executable,
                            str(REPO / "scripts/run-web-coding-benchmark.py"),
                            "--fixture",
                            str(a.fixture.resolve()),
                            "--output",
                            str((a.output / engine).resolve()),
                            "--base",
                            base,
                            "--container",
                            container,
                            "--max-wall-seconds",
                            str(a.max_wall_seconds),
                            "--thinking-token-budget",
                            str(a.thinking_token_budget),
                        ],
                        cwd=REPO,
                        timeout=a.max_wall_seconds + 1800,
                    )
                except subprocess.CalledProcessError:
                    result_path = a.output / engine / "summary.json"
                    if not result_path.exists():
                        raise
                    stopped = json.loads(result_path.read_text())
                    reason = str(stopped.get("error", ""))
                    if stopped["status"] != "failed" or not (
                        reason.startswith(
                            "RuntimeError: common context window exhausted:"
                        )
                        or reason == "RuntimeError: run wall-time limit"
                    ):
                        raise
                    # Exhausting a declared task budget is an agent outcome.
                    # Preserve the failed session and run the other same task.
                measured = json.loads((a.output / engine / "summary.json").read_text())
                state.setdefault("engine_outcomes", {})[engine] = {
                    "session_status": measured["status"],
                    "stop_reason": measured.get("error"),
                }
                R.save(a.output / "campaign.json", state)
                if engine == "gptq" and "exl3" in engines:
                    measured = json.loads(
                        (a.output / engine / "summary.json").read_text()
                    )
                    minimum = json.loads((a.fixture / "manifest.json").read_text()).get(
                        "target_context_min", 100000
                    )
                    if measured["overall"]["context_max"] < minimum:
                        state.update(
                            status="needs_longer_fixture",
                            context_max=measured["overall"]["context_max"],
                            finished_at=R.now(),
                        )
                        R.save(a.output / "campaign.json", state)
                        return
                if engine == "exl3":
                    logs = subprocess.check_output(
                        ["docker", "logs", container],
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    (a.output / "exl3-server.log").write_text(logs)
                    stop_exl()
            state.update(status="complete", finished_at=R.now())
            R.save(a.output / "campaign.json", state)
        except BaseException as e:
            state.update(
                status="failed",
                error=type(e).__name__ + ": " + str(e),
                finished_at=R.now(),
            )
            R.save(a.output / "campaign.json", state)
            raise
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            if a.restore_after_campaign:
                R.save(a.output / "production-restored.json", restore())
            else:
                stop_exl()
                command(["systemctl", "--user", "stop", SERVICE])
                R.save(a.output / "production-left-offline.json", {"unix": time.time(), "user_requested": True})


if __name__ == "__main__":
    main()

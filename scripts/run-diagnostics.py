#!/usr/bin/env python3
"""Run M01 long-context observability and a pre-capture full decode trace."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import urllib.request


REPO = Path(__file__).resolve().parents[1]
PRODUCTION_CONTAINER = "qwen38-vllm-production"
PRODUCTION_SERVICE = "qwen38.service"
PRODUCTION_URL = "http://127.0.0.1:8081"
EXPERIMENT_URL = "http://127.0.0.1:18087"
EXPECTED_IMAGE_ID = "sha256:3f20b0cf493fe0904a7efd0ca310067790e901bc5d60203340c411e57c25010a"
PROFILE_IMAGE = "local/qwen38-b70-vllm:m01-profile-before-capture"
LONG_REQUEST = Path(
    "/home/sebastian/LocalLLM/Local-AI-B70/qwen38/optimization/"
    "q128-full-context-208k/runs/run-20260914-215108/request.json"
)
DECODE_FIXTURE = Path(
    "/home/sebastian/LocalLLM/Local-AI-B70/qwen38/optimization/"
    "power-decode-study/fixtures/8192.json"
)
METRIC_PREFIXES = (
    "vllm:num_preemptions_total",
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:request_prefill_kv_computed_tokens",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)


def api(path: str, data=None, timeout: int = 30, base: str = EXPERIMENT_URL):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        return json.loads(body) if body else None


def prometheus(base: str = EXPERIMENT_URL) -> tuple[dict[str, float], str]:
    raw = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()
    result = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 2:
            try:
                result[fields[0]] = float(fields[-1])
            except ValueError:
                pass
    return result, raw


def metric_total(metrics: dict[str, float], name: str) -> float:
    return sum(value for key, value in metrics.items() if key.split("{")[0] == name)


def relevant_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        key: value
        for key, value in metrics.items()
        if any(key.split("{")[0].startswith(prefix) for prefix in METRIC_PREFIXES)
    }


def one_path(pattern: str) -> Path:
    paths = list(Path("/").glob(pattern.lstrip("/")))
    if len(paths) != 1:
        raise RuntimeError(f"expected one path for {pattern}, found {len(paths)}")
    return paths[0]


POWER_CAP = one_path("/sys/bus/pci/devices/0000:03:00.0/hwmon/*/power1_cap")
ENERGY = POWER_CAP.with_name("energy1_input")


def save(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def production_inspect() -> dict:
    return json.loads(
        subprocess.check_output(["docker", "inspect", PRODUCTION_CONTAINER], text=True)
    )[0]


def assert_mtp4(command: list[str]) -> None:
    position = command.index("--speculative-config")
    spec = json.loads(command[position + 1])
    if spec != {"method": "mtp", "num_speculative_tokens": 4}:
        raise RuntimeError(f"MTP4 control changed: {spec}")


def ensure_idle(base: str) -> None:
    metrics, _ = prometheus(base)
    for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        if metric_total(metrics, name) != 0:
            raise RuntimeError(f"server is busy: {name}={metric_total(metrics, name)}")


def engine_command(
    inspect: dict,
    *,
    name: str,
    image: str,
    evidence: Path,
    arguments: list[str],
    profile_before_capture: bool = False,
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--device",
        "/dev/dri",
        "--group-add",
        str(Path("/dev/dri/renderD128").stat().st_gid),
        "-v",
        "/dev/dri:/dev/dri:ro",
        "--shm-size",
        "8g",
        "-p",
        "127.0.0.1:18087:8000",
        "-v",
        f"{evidence}:/evidence",
    ]
    for mount in inspect["Mounts"]:
        if mount["Destination"] == "/dev/dri":
            continue
        suffix = "" if mount["RW"] else ":ro"
        command += ["-v", f"{mount['Source']}:{mount['Destination']}{suffix}"]
    environment = inspect["Config"].get("Env") or []
    for row in environment:
        if row.startswith(("B70_", "VLLM_", "ZE_", "PYTORCH_", "HF_HUB_OFFLINE=", "PYTHONPATH=")):
            command += ["-e", row]
    if profile_before_capture:
        command += ["-e", "B70_PROFILE_BEFORE_CAPTURE=1"]
    command += [
        "-v",
        f"{REPO / 'scripts/memory_sampler.py'}:/opt/b70-diagnostics/memory_sampler.py:ro",
        "--entrypoint",
        "vllm",
        image,
        *arguments,
    ]
    return command


def wait_ready(process: subprocess.Popen, timeout: int = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"engine exited during startup with status {process.returncode}")
        try:
            api("/health", timeout=2)
            return
        except OSError:
            time.sleep(2)
    raise TimeoutError("diagnostic engine readiness")


def stop_container(name: str, process: subprocess.Popen | None) -> None:
    subprocess.run(
        ["docker", "stop", "-t", "30", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=45,
    )
    if process is not None:
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


class TelemetrySampler:
    def __init__(self, path: Path):
        self.path = path
        self.stop = threading.Event()
        self.errors: list[str] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        with self.path.open("x") as output:
            while not self.stop.is_set():
                row = {
                    "time": time.time(),
                    "monotonic_ns": time.monotonic_ns(),
                    "energy_uj": int(ENERGY.read_text()),
                    "power_cap_uw": int(POWER_CAP.read_text()),
                    "temperatures_millic": {
                        path.name: int(path.read_text())
                        for path in POWER_CAP.parent.glob("temp*_input")
                    },
                }
                try:
                    metrics, _ = prometheus()
                    row["metrics"] = relevant_metrics(metrics)
                except Exception as error:
                    row["metrics_error"] = repr(error)
                    self.errors.append(repr(error))
                output.write(json.dumps(row) + "\n")
                output.flush()
                self.stop.wait(0.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join(timeout=5)


def stream_request(payload: dict, path: Path) -> dict:
    request = urllib.request.Request(
        EXPERIMENT_URL + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    first = None
    usage = None
    finish_reasons = []
    response_parts = []
    with path.open("x") as output, urllib.request.urlopen(request, timeout=2_400) as response:
        for encoded in response:
            now = time.monotonic()
            line = encoded.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            text = line[5:].strip()
            output.write(json.dumps({"elapsed_s": now - start, "payload": text}) + "\n")
            output.flush()
            if text == "[DONE]":
                break
            event = json.loads(text)
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                for name in ("content", "reasoning", "reasoning_content"):
                    if delta.get(name):
                        response_parts.append(delta[name])
                        break
                if first is None and any(
                    delta.get(name) for name in ("content", "reasoning", "reasoning_content")
                ):
                    first = now
                if choice.get("finish_reason"):
                    finish_reasons.append(choice["finish_reason"])
    end = time.monotonic()
    if usage is None:
        raise RuntimeError("long-context response had no usage record")
    return {
        "usage": usage,
        "wall_s": end - start,
        "ttft_s": first - start if first is not None else None,
        "finish_reasons": finish_reasons,
        "response_text": "".join(response_parts),
    }


ITERATION = re.compile(
    r"INFO (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d).*Iteration\((\d+)\): "
    r"(\d+) context requests, (\d+) context tokens, (\d+) generation requests, "
    r"(\d+) generation tokens, iteration elapsed time: ([0-9.]+) ms, "
    r"GPU KV cache usage: ([0-9.]+)%"
)


def parse_iterations(log: str) -> list[dict]:
    rows = []
    cumulative = 0
    year = dt.datetime.now(dt.timezone.utc).year
    for match in ITERATION.finditer(log):
        month, day, hour, minute, second = map(int, match.group(1, 2, 3, 4, 5))
        context_tokens = int(match.group(8))
        cumulative += context_tokens
        rows.append(
            {
                "time_unix": dt.datetime(
                    year, month, day, hour, minute, second, tzinfo=dt.timezone.utc
                ).timestamp(),
                "iteration": int(match.group(6)),
                "context_requests": int(match.group(7)),
                "context_tokens": context_tokens,
                "generation_requests": int(match.group(9)),
                "generation_tokens": int(match.group(10)),
                "elapsed_ms": float(match.group(11)),
                "kv_cache_usage_pct": float(match.group(12)),
                "cumulative_scheduled_context_tokens": cumulative,
            }
        )
    return rows


def telemetry_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def preemption_metric(metrics: dict[str, float]) -> float:
    return metric_total(metrics, "vllm:num_preemptions_total")


def memory_summary(path: Path) -> dict:
    samples = [json.loads(line) for line in path.read_text().splitlines()]
    peaks: dict[str, int] = {}
    model_pids = set()
    for sample in samples:
        for client in sample["clients"].values():
            model_pids.add(client["pid"])
            for name, value in client.items():
                if not name.startswith("drm-"):
                    continue
                match = re.match(r"(\d+)\s+KiB$", str(value))
                if match:
                    peaks[name] = max(peaks.get(name, 0), int(match.group(1)))
    return {"sample_count": len(samples), "client_pids": sorted(model_pids), "peak_kib": peaks}


def scheduler_summary(engine_log: Path, telemetry: Path, before: dict, after: dict) -> dict:
    iterations = parse_iterations(engine_log.read_text())
    context_rows = [row for row in iterations if row["context_tokens"]]
    scheduled = context_rows[-1]["cumulative_scheduled_context_tokens"] if context_rows else 0
    checkpoints = []
    for target in (98_304, 131_072, 196_608, 200_448):
        row = next(
            (item for item in context_rows if item["cumulative_scheduled_context_tokens"] >= target),
            None,
        )
        if row:
            checkpoints.append({"target": target, **row})
    log_preemptions = []
    for previous, current in zip(iterations, iterations[1:]):
        if previous["kv_cache_usage_pct"] >= 90 and current["kv_cache_usage_pct"] == 0:
            log_preemptions.append(
                {
                    "time_unix": current["time_unix"],
                    "iteration": current["iteration"],
                    "cache_before_pct": previous["kv_cache_usage_pct"],
                    "stall_elapsed_ms": current["elapsed_ms"],
                    "scheduled_tokens_at_event": current["cumulative_scheduled_context_tokens"],
                }
            )
    samples = telemetry_rows(telemetry)
    metric_preemptions = []
    last = None
    for sample in samples:
        metrics = sample.get("metrics") or {}
        value = preemption_metric(metrics)
        if last is not None and value > last:
            metric_preemptions.append(
                {"time_unix": sample["time"], "counter_before": last, "counter_after": value}
            )
        last = value
    delta = {
        name: metric_total(after, name) - metric_total(before, name)
        for name in (
            "vllm:num_preemptions_total",
            "vllm:request_prefill_kv_computed_tokens_sum",
            "vllm:request_prefill_time_seconds_sum",
            "vllm:request_decode_time_seconds_sum",
            "vllm:prompt_tokens_total",
            "vllm:generation_tokens_total",
        )
    }
    return {
        "iteration_count": len(iterations),
        "scheduled_prefill_tokens": scheduled,
        "expected_prompt_tokens": 200_448,
        "recomputed_scheduler_tokens": scheduled - 200_448,
        "scheduler_work_ratio": scheduled / 200_448 if scheduled else None,
        "metric_delta": delta,
        "checkpoints": checkpoints,
        "log_inferred_preemption_events": log_preemptions,
        "metric_counter_preemption_events": metric_preemptions,
        "telemetry_sample_count": len(samples),
        "peak_kv_cache_usage_metric": max(
            (
                metric_total(row.get("metrics") or {}, "vllm:kv_cache_usage_perc")
                for row in samples
            ),
            default=None,
        ),
    }


def run_long_context(
    inspect: dict,
    run_dir: Path,
    *,
    phase_name: str = "long-context",
    container_name: str = "b70-m01-long-context",
    extra_arguments: tuple[str, ...] = (),
    image: str | None = None,
    payload_overrides: dict | None = None,
) -> dict:
    phase = run_dir / phase_name
    phase.mkdir()
    payload = json.loads(LONG_REQUEST.read_text())
    if payload_overrides:
        payload.update(payload_overrides)
    (phase / "request.json").write_text(json.dumps(payload, separators=(",", ":")))
    arguments = inspect["Config"]["Cmd"].copy()
    assert_mtp4(arguments)
    arguments += [*extra_arguments, "--enable-logging-iteration-details"]
    command = engine_command(
        inspect,
        name=container_name,
        image=image or inspect["Image"],
        evidence=phase,
        arguments=arguments,
    )
    save(phase / "engine-command.json", command)
    process = None
    log_handle = None
    try:
        log_handle = (phase / "engine.log").open("w")
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
        wait_ready(process)
        startup, raw = prometheus()
        (phase / "startup-metrics.txt").write_text(raw)
        capacity_match = re.search(r'kv_cache_size_tokens="(\d+)"', raw)
        capacity = int(capacity_match.group(1)) if capacity_match else None
        if capacity is None or capacity < 200_704:
            raise RuntimeError(f"insufficient KV capacity: {capacity}")
        before, raw = prometheus()
        (phase / "before-metrics.txt").write_text(raw)
        if metric_total(before, "vllm:prefix_cache_hits_total") != 0:
            raise RuntimeError("fresh diagnostic engine unexpectedly reports prefix-cache hits")
        energy_start = int(ENERGY.read_text())
        subprocess.run(
            [
                "docker",
                "exec",
                "-d",
                container_name,
                "python",
                "/opt/b70-diagnostics/memory_sampler.py",
            ],
            check=True,
        )
        with TelemetrySampler(phase / "telemetry.jsonl") as sampler:
            response = stream_request(payload, phase / "response.sse.jsonl")
        (phase / "stop-memory-sampler").touch()
        time.sleep(1)
        after, raw = prometheus()
        (phase / "after-metrics.txt").write_text(raw)
        response["energy_j"] = (int(ENERGY.read_text()) - energy_start) / 1e6
        response["kv_capacity_tokens"] = capacity
        response["request_sha256"] = sha256(LONG_REQUEST)
        response["telemetry_errors"] = sampler.errors
        if response["usage"].get("prompt_tokens") != 200_448:
            raise RuntimeError(f"prompt-token mismatch: {response['usage']}")
    finally:
        if not (phase / "stop-memory-sampler").exists():
            (phase / "stop-memory-sampler").touch()
        stop_container(container_name, process)
        if log_handle:
            log_handle.close()
    scheduler = scheduler_summary(phase / "engine.log", phase / "telemetry.jsonl", before, after)
    result = {**response, "scheduler": scheduler, "memory": memory_summary(phase / "memory-samples.jsonl")}
    save(phase / "summary.json", result)
    return result


def run_decode_trace(inspect: dict, run_dir: Path) -> dict:
    phase = run_dir / "decode-trace"
    traces = phase / "traces"
    traces.mkdir(parents=True)
    fixture = json.loads(DECODE_FIXTURE.read_text())
    payload = {
        "model": "Qwen3.8-27B",
        "messages": fixture["messages"],
        "max_tokens": 256,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
        "thinking_token_budget": 8192,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": "medium",
        },
    }
    save(phase / "request.json", payload)
    profile = {
        "profiler": "torch",
        "torch_profiler_dir": "/evidence/traces",
        "torch_profiler_with_stack": False,
        "torch_profiler_record_shapes": True,
        "torch_profiler_dump_cuda_time_total": False,
        "torch_profiler_with_memory": False,
        "ignore_frontend": True,
        "delay_iterations": 0,
        "max_iterations": 0,
        "detailed_trace_annotation": True,
    }
    arguments = inspect["Config"]["Cmd"].copy()
    assert_mtp4(arguments)
    arguments += ["--profiler-config", json.dumps(profile, separators=(",", ":"))]
    command = engine_command(
        inspect,
        name="b70-m01-decode-trace",
        image=PROFILE_IMAGE,
        evidence=phase,
        arguments=arguments,
        profile_before_capture=True,
    )
    save(phase / "engine-command.json", command)
    process = None
    log_handle = None
    try:
        log_handle = (phase / "engine.log").open("w")
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
        wait_ready(process)
        started = time.monotonic()
        response = api("/v1/chat/completions", payload, timeout=1_800)
        elapsed = time.monotonic() - started
        api("/stop_profile", {}, timeout=600)
        save(phase / "response.json", response)
        usage = response.get("usage") or {}
        if usage.get("prompt_tokens") != 8192 or usage.get("completion_tokens") != 256:
            raise RuntimeError(f"decode trace token mismatch: {usage}")
        result = {
            "wall_s_with_profiler": elapsed,
            "usage": usage,
            "profile_config": profile,
            "fixture_sha256": sha256(DECODE_FIXTURE),
            "prompt_sha256": fixture.get("prompt_sha256"),
            "profiler_started_before_graph_capture": True,
            "mtp_depth": 4,
        }
    finally:
        stop_container("b70-m01-decode-trace", process)
        if log_handle:
            log_handle.close()
    trace_files = sorted(traces.rglob("*.pt.trace.json.gz"))
    if len(trace_files) != 1:
        raise RuntimeError(f"expected one trace file, found {len(trace_files)}")
    result["trace_file"] = str(trace_files[0])
    result["trace_size_bytes"] = trace_files[0].stat().st_size
    save(phase / "summary.json", result)
    subprocess.run(
        ["python3", str(REPO / "scripts/analyze-diagnostics.py"), str(run_dir)],
        check=True,
        timeout=600,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "preflight",
        "started_at_unix": time.time(),
        "run_dir": str(run_dir),
        "phases": {},
        "mtp_depth": 4,
    }
    state_path = run_dir / "state.json"
    save(state_path, state)
    lock_path = Path(
        "/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    inspect = production_inspect()
    if inspect["Image"] != EXPECTED_IMAGE_ID:
        raise RuntimeError(f"production image identity changed: {inspect['Image']}")
    assert_mtp4(inspect["Config"]["Cmd"])
    if int(POWER_CAP.read_text()) != 180_000_000:
        raise RuntimeError("power cap is not 180 W")
    ensure_idle(PRODUCTION_URL)
    subprocess.run(
        [
            "python3",
            str(REPO / "scripts/collect-runtime-manifest.py"),
            "--container",
            PRODUCTION_CONTAINER,
            "--output",
            str(run_dir / "manifest.json"),
        ],
        check=True,
        timeout=60,
        cwd=REPO,
    )
    profile_inspect = json.loads(
        subprocess.check_output(["docker", "image", "inspect", PROFILE_IMAGE], text=True)
    )[0]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["diagnostic_profile_image"] = {
        "reference": PROFILE_IMAGE,
        "image_id": profile_inspect["Id"],
        "base_control_image_id": EXPECTED_IMAGE_ID,
        "dockerfile_sha256": sha256(REPO / "docker/diagnostics/Dockerfile.profile"),
        "patch_sha256": sha256(REPO / "docker/diagnostics/patch_profile_before_capture.py"),
    }
    manifest["frozen_requests"] = {
        "long_context_sha256": sha256(LONG_REQUEST),
        "decode_fixture_sha256": sha256(DECODE_FIXTURE),
    }
    save(run_dir / "manifest.json", manifest)
    save(run_dir / "production-inspect.json", inspect)
    state["status"] = "stopping-production"
    save(state_path, state)

    try:
        subprocess.run(
            ["systemctl", "--user", "stop", PRODUCTION_SERVICE], check=True, timeout=120
        )
        state["status"] = "long-context"
        save(state_path, state)
        state["phases"]["long_context"] = run_long_context(inspect, run_dir)
        save(state_path, state)
        state["status"] = "decode-trace"
        save(state_path, state)
        state["phases"]["decode_trace"] = run_decode_trace(inspect, run_dir)
        state["status"] = "completed"
        state["finished_at_unix"] = time.time()
        save(state_path, state)
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        state["finished_at_unix"] = time.time()
        save(state_path, state)
        raise
    finally:
        subprocess.run(
            ["python3", str(REPO / "scripts/restore-production.py")],
            check=True,
            timeout=800,
            env={**os.environ, "B70_RUN_DIR": str(run_dir)},
        )

    decision = [
        "# M01 observability decision",
        "",
        "Status: measurement completed; this run changes no production configuration.",
        "",
        "MTP remained fixed at four. The long-context phase attributes scheduler work,",
        "preemption timing, VRAM and power. The decode phase records graph capture and",
        "all subsequent context=0/generation=1 execution windows for cost attribution.",
        "",
        "Promotion decision: none; diagnostic evidence only.",
        "",
        "Rollback: `systemctl --user start qwen38.service`.",
    ]
    (run_dir / "decision.md").write_text("\n".join(decision) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

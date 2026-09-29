#!/usr/bin/env python3
"""Execute one frozen C3 serving worker slice on the live candidate."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.request


REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "benchmark-results/production-release-v1/serving-fixtures"
SCHEDULE = REPO / "config/integrated_serving_schedule_v1.json"
IMAGE = "sha256:a42cda993bf6492acc39d23e9382e27a17efca4657bea07d80d0c28228a1623a"
VRAM_USED = Path("/sys/class/drm/card1/device/mem_info_vram_used")
METRICS = {
    "prefill_tokens": "vllm:request_prefill_kv_computed_tokens_sum",
    "prefill_s": "vllm:request_prefill_time_seconds_sum",
    "decode_s": "vllm:request_decode_time_seconds_sum",
    "logical_prompt_tokens": "vllm:prompt_tokens_total",
    "cached_tokens": "vllm:prompt_tokens_cached_total",
    "generated_tokens": "vllm:generation_tokens_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "preemptions": "vllm:num_preemptions_total",
    "queue_s": "vllm:request_queue_time_seconds_sum",
}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def identity(container: str) -> dict:
    item = json.loads(subprocess.check_output(
        ["docker", "inspect", container], text=True))[0]
    if item["Image"] != IMAGE:
        raise RuntimeError(f"wrong serving image {item['Image']}")
    return {"container_id": item["Id"], "image_id": item["Image"],
            "command": item["Config"]["Cmd"],
            "policy_sha256": item["Config"]["Labels"].get(
                "org.local.b70.policy.sha256")}


def verify_inputs() -> dict:
    manifest = json.loads((REPO / "config/frozen_fixture_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest((FIXTURES / name).read_bytes()) != expected:
            raise RuntimeError(f"serving fixture changed: {name}")
    schedule = json.loads(SCHEDULE.read_text())
    if schedule["counts"] != {"ordinary": 12, "mixed": 6, "unequal": 8,
                              "capacity": 4, "cancel": 6, "prefix": 4}:
        raise RuntimeError("wrong serving schedule counts")
    actual = {key: 0 for key in schedule["counts"]}
    for worker in schedule["workers"]:
        for group in worker["groups"]:
            actual[group["kind"]] += (
                1 + len(group["recovery"]) if group["kind"] == "cancel"
                else len(group["requests"]))
    if actual != schedule["counts"] or sum(actual.values()) != 40:
        raise RuntimeError(f"serving schedule count mismatch: {actual}")
    return schedule


def payload(spec: dict) -> tuple[dict, str]:
    case, repeat = spec["case"], spec["repeat"]
    folder = (FIXTURES / ("full-context-199680-r1" if case == "199k"
                          else f"phase-{case}-c1-r{repeat}"))
    frozen = json.loads((folder / "request-1.json").read_text())
    prompt = (folder / "prompt-1.txt").read_text()
    if digest(prompt.encode()) != frozen["prompt_sha256"]:
        raise RuntimeError(f"frozen prompt changed: {folder}")
    if spec.get("variant") == "modified":
        prompt += "\n\nModified-prefix check: include the new suffix in your review.\n"
    elif spec.get("variant", "exact") != "exact":
        raise ValueError("unknown prefix variant")
    body = {key: frozen[key] for key in (
        "model", "max_tokens", "temperature", "top_p", "top_k", "seed",
        "ignore_eos", "chat_template_kwargs", "stream", "stream_options")}
    body["messages"] = [{"role": "user", "content": prompt}]
    return body, digest(prompt.encode())


def prometheus(base: str):
    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
        raw = response.read().decode()
    result = {key: 0.0 for key in METRICS}
    result.update({"abort": 0.0, "running": 0.0, "waiting": 0.0,
                   "kv_usage": 0.0})
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([^\s{]+)(?:\{([^}]*)\})?\s+([0-9.eE+-]+)$", line)
        if not match:
            continue
        name, labels, value = match.group(1), match.group(2) or "", float(match.group(3))
        for key, metric in METRICS.items():
            if name == metric:
                result[key] += value
        if name == "vllm:request_success_total" and 'finished_reason="abort"' in labels:
            result["abort"] += value
        for key, metric in (("running", "vllm:num_requests_running"),
                            ("waiting", "vllm:num_requests_waiting"),
                            ("kv_usage", "vllm:kv_cache_usage_perc")):
            if name == metric:
                result[key] += value
    return result


def quantile(values: list[float], fraction: float):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def stream(base: str, body: dict, barrier: threading.Barrier | None = None):
    request = urllib.request.Request(base + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    if barrier:
        barrier.wait()
    started = time.monotonic()
    first = None
    pieces = []
    stamps = []
    usage = None
    finish = None
    with urllib.request.urlopen(request, timeout=1500) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                part = delta.get("content") or delta.get("reasoning_content") or ""
                if part:
                    moment = time.monotonic()
                    if first is None:
                        first = moment
                    pieces.append(part)
                    stamps.append(moment)
                finish = choice.get("finish_reason") or finish
    ended = time.monotonic()
    gaps = [right - left for left, right in zip(stamps, stamps[1:])]
    return {"start_monotonic": started, "end_monotonic": ended,
            "ttft_s": first - started if first else None,
            "wall_s": ended - started, "first_piece_monotonic": first,
            "stream_piece_times_monotonic": stamps,
            "bundle_gap_p50_s": statistics.median(gaps) if gaps else None,
            "bundle_gap_p95_s": quantile(gaps, .95),
            "bundle_gap_max_s": max(gaps) if gaps else None,
            "stream_piece_count": len(stamps),
            "usage": usage, "finish_reason": finish,
            "output_sha256": digest("".join(pieces).encode())}


def sample_memory(base: str, stop: threading.Event, output: Path):
    with output.open("w") as handle:
        while not stop.is_set():
            row = {"monotonic": time.monotonic(),
                   "vram_used_bytes": int(VRAM_USED.read_text())}
            try:
                metrics = prometheus(base)
                row.update({key: metrics[key] for key in (
                    "running", "waiting", "kv_usage", "generated_tokens")})
            except OSError as error:
                row["sample_error"] = repr(error)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            stop.wait(2)


def group_run(base: str, group: dict, output: Path):
    before = prometheus(base)
    stop = threading.Event()
    sample_path = output / f"{group['id']}-memory.jsonl"
    sampler = threading.Thread(target=sample_memory,
        args=(base, stop, sample_path), daemon=True)
    sampler.start()
    started = time.monotonic()
    requests = []
    try:
        if group["kind"] == "cancel":
            body, prompt_hash = payload(group["cancel"])
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as temporary:
                json.dump(body, temporary)
                temporary.flush()
                process = subprocess.Popen([
                    "curl", "--no-buffer", "-sS", "-o", "/dev/null",
                    "-H", "Content-Type: application/json", "--data-binary",
                    "@" + temporary.name, base + "/v1/chat/completions"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                cancel_start = time.monotonic()
                try:
                    deadline = cancel_start + 30
                    while time.monotonic() < deadline and prometheus(base)["running"] < 1:
                        if process.poll() is not None:
                            raise RuntimeError("cancellation request ended before admission")
                        time.sleep(.25)
                    if prometheus(base)["running"] < 1:
                        raise RuntimeError("cancellation prefill was never admitted")
                    time.sleep(2)
                finally:
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=20)
                requests.append({"case": group["cancel"], "prompt_sha256": prompt_hash,
                                 "cancelled": True, "wall_s": time.monotonic() - cancel_start,
                                 "client_exit_code": process.returncode})
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline and prometheus(base)["abort"] <= before["abort"]:
                time.sleep(1)
            if prometheus(base)["abort"] <= before["abort"]:
                raise RuntimeError("server did not record a cancelled request")
            specs = group["recovery"]
        else:
            specs = group["requests"]
        if group["kind"] == "prefix":
            for spec in specs:
                body, prompt_hash = payload(spec)
                metric_before = prometheus(base)
                result = stream(base, body)
                metric_after = prometheus(base)
                requests.append({"case": spec, "prompt_sha256": prompt_hash,
                                 "metrics": {key: metric_after[key] - metric_before[key]
                                             for key in METRICS}, **result})
        else:
            barrier = threading.Barrier(len(specs))
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as pool:
                futures = []
                for spec in specs:
                    body, prompt_hash = payload(spec)
                    futures.append((spec, prompt_hash,
                                    pool.submit(stream, base, body, barrier)))
                for spec, prompt_hash, future in futures:
                    requests.append({"case": spec, "prompt_sha256": prompt_hash,
                                     **future.result(timeout=1600)})
    finally:
        stop.set()
        sampler.join(timeout=10)
    after = prometheus(base)
    metrics = {key: after[key] - before[key] for key in METRICS}
    metrics["abort"] = after["abort"] - before["abort"]
    memory = [json.loads(line) for line in sample_path.read_text().splitlines()]
    completed = [row for row in requests if not row.get("cancelled")]
    failures = []
    if metrics["preemptions"]:
        failures.append("preemption")
    if group["kind"] == "cancel" and metrics["abort"] < 1:
        failures.append("cancellation_not_recorded")
    for row in completed:
        if ((row.get("usage") or {}).get("completion_tokens") != 1024
                or row.get("finish_reason") != "length"):
            failures.append(f"incomplete_output:{row['case']}")
    if group["kind"] == "prefix" and len(requests) == 2:
        if requests[1]["metrics"]["cached_tokens"] <= 0:
            failures.append("prefix_not_reused")
    overlap = (max((row["first_piece_monotonic"] for row in completed), default=None),
               min((row["end_monotonic"] for row in completed), default=None))
    overlap_tokens = None
    if overlap[0] and overlap[1] and overlap[1] > overlap[0]:
        left = min(memory, key=lambda row: abs(row["monotonic"] - overlap[0]))
        right = min(memory, key=lambda row: abs(row["monotonic"] - overlap[1]))
        overlap_tokens = right.get("generated_tokens", 0) - left.get("generated_tokens", 0)
    return {"id": group["id"], "kind": group["kind"],
            "start_monotonic": started, "wall_s": time.monotonic() - started,
            "requests": requests, "metrics": metrics,
            "vram_baseline_bytes": memory[0]["vram_used_bytes"] if memory else None,
            "vram_peak_bytes": max((row["vram_used_bytes"] for row in memory), default=None),
            "kv_peak": max((row.get("kv_usage", 0) for row in memory), default=None),
            "overlap_window_s": max(0, overlap[1] - overlap[0])
                                if overlap[0] and overlap[1] else None,
            "generated_tokens_in_overlap_window": overlap_tokens,
            "failures": failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--container", default="b70-qwen38-vllm")
    args = parser.parse_args()
    schedule = verify_inputs()
    worker = next((row for row in schedule["workers"] if row["index"] == args.worker_index), None)
    if worker is None:
        raise ValueError("worker index not in frozen schedule")
    if args.output.exists():
        raise ValueError("output directory exists")
    live = identity(args.container)
    args.output.mkdir(parents=True)
    (args.output / "identity.json").write_text(json.dumps({
        "worker_index": args.worker_index, "live": live,
        "schedule_sha256": digest(SCHEDULE.read_bytes())}, indent=2) + "\n")
    for group in worker["groups"]:
        if identity(args.container)["container_id"] != live["container_id"]:
            raise RuntimeError("worker changed during serving trace")
        result = group_run(args.base, group, args.output)
        (args.output / f"{group['id']}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"group": group["id"], "wall_s": round(result["wall_s"], 2),
                          "requests": len(result["requests"]),
                          "failures": result["failures"]}), flush=True)
    print(json.dumps({"worker_index": args.worker_index, "state": "complete"}), flush=True)


if __name__ == "__main__":
    main()

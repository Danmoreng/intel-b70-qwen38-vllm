#!/usr/bin/env python3
"""Paired unequal-context C2 serving screen with fixed continuations."""

import argparse
import json
import re
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from replay_frozen import METRICS, snapshot
from staggered_serving import payload, stream


GAUGES = ("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc")
IMAGE = "local/b70-qwen38-vllm:onednn-poc-20260929"


def cache_gauges(base):
    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()
    values = {}
    for line in lines:
        match = re.match(r"([^\s{]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if match and match.group(1) in GAUGES:
            values[match.group(1)] = max(
                values.get(match.group(1), 0.0), float(match.group(2)))
    return values


def quantile(values, fraction):
    if not values:
        return None
    sorted_values = sorted(values)
    return sorted_values[min(len(values) - 1, int(fraction * (len(values) - 1)))]


def describe_request(result):
    stamps = result["stream_piece_times_s"]
    gaps = [right - left for left, right in zip(stamps, stamps[1:])]
    return {
        "start_s": result["start_s"],
        "ttft_s": stamps[0] - result["start_s"],
        "wall_s": result["end_s"] - result["start_s"],
        "usage": result["usage"],
        "finish_reason": result["finish_reason"],
        "output_sha256": result["output_sha256"],
        "stream_piece_count": result["stream_piece_count"],
        "piece_gap_p50_s": statistics.median(gaps) if gaps else None,
        "piece_gap_p95_s": quantile(gaps, 0.95),
        "piece_gap_p99_s": quantile(gaps, 0.99),
        "piece_gap_max_s": max(gaps) if gaps else None,
        "stream_piece_times_s": stamps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--short-case", default="32k")
    parser.add_argument("--long-case", default="128k")
    parser.add_argument("--forced-token-id", type=int, default=264)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.short_case == args.long_case:
        parser.error("use two different frozen source contexts")
    bodies = {}
    hashes = {}
    for label, case in (("short", args.short_case), ("long", args.long_case)):
        body, digest = payload(case)
        body["allowed_token_ids"] = [args.forced_token_id]
        body["temperature"] = 0
        bodies[label], hashes[label] = body, digest

    before = snapshot(args.base)
    started = time.monotonic()
    results = {label: {"first_piece": threading.Event()} for label in bodies}
    threads = [threading.Thread(target=stream,
                args=(args.base, bodies[label], results[label], started), daemon=True)
               for label in ("short", "long")]
    gauge_samples = []
    stop_polling = threading.Event()

    def poll_gauges():
        while not stop_polling.is_set():
            try:
                gauge_samples.append({"at_s": time.monotonic() - started,
                                      **cache_gauges(args.base)})
            except Exception:
                pass
            stop_polling.wait(2)

    poller = threading.Thread(target=poll_gauges, daemon=True)
    poller.start()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=900)
    stop_polling.set()
    poller.join(timeout=10)
    if any(thread.is_alive() for thread in threads):
        raise TimeoutError("concurrent request did not finish")
    if any("error" in row for row in results.values()):
        raise RuntimeError({label: row.get("error") for label, row in results.items()})
    after = snapshot(args.base)
    metrics = {key: after[key] - before[key] for key in METRICS}
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
        text=True).strip()
    described = {label: describe_request(row) for label, row in results.items()}
    peaks = {name: max((row[name] for row in gauge_samples if name in row),
                       default=None)
             for name in GAUGES}
    result = {
        "arm": args.arm, "image_id": image_id,
        "cases": {"short": args.short_case, "long": args.long_case},
        "prompt_sha256": hashes, "forced_token_id": args.forced_token_id,
        "metrics": metrics, "cache_gauge_peaks": peaks,
        "cache_gauge_samples": gauge_samples,
        "requests": described,
        "start_skew_s": abs(described["short"]["start_s"] -
                            described["long"]["start_s"]),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("cache_gauge_samples", "requests")}),
          flush=True)
    if metrics["cached_tokens"] or metrics["preemptions"]:
        raise RuntimeError("unequal C2 pair had cache hit or preemption")
    for label, row in described.items():
        if row["usage"]["completion_tokens"] != 1024:
            raise RuntimeError(f"{label} did not emit 1024 tokens")


if __name__ == "__main__":
    main()

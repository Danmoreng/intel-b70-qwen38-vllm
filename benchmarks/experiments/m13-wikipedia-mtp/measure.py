#!/usr/bin/env python3
"""Measure seeded Wikipedia summaries on an already-running MTP server."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location(
    "b70_profile", REPO / "scripts/current-profile-benchmark.py"
)
assert spec and spec.loader
profile = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = profile
spec.loader.exec_module(profile)

MODEL = "Qwen3.8-27B"
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512}


def request_seed(concurrency: int, repeat: int, article_index: int, position: int) -> int:
    return 770000 + concurrency * 1000 + repeat * 100 + article_index * 10 + position


def article_indices(concurrency: int, repeat: int) -> list[int]:
    if concurrency in (1, 2):
        return [(repeat * concurrency + position) % 4 for position in range(concurrency)]
    return [(repeat + position) % 4 for position in range(concurrency)]


def stream_chat(base: str, payload: dict, barrier: threading.Barrier,
                stream_path: Path, response_path: Path) -> dict:
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    barrier.wait()
    start = time.monotonic()
    first = None
    usage = None
    finish_reason = None
    content: list[str] = []
    reasoning: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=300) as response, stream_path.open("w") as stream:
            for raw in response:
                if not raw.startswith(b"data: ") or raw.strip() == b"data: [DONE]":
                    continue
                event = json.loads(raw[6:])
                elapsed = time.monotonic() - start
                stream.write(json.dumps({"elapsed_s": elapsed, "event": event}, ensure_ascii=False) + "\n")
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if piece and first is None:
                        first = elapsed
                    content.append(piece)
                    reasoning.append(thought)
                    finish_reason = choice.get("finish_reason") or finish_reason
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error
    wall = time.monotonic() - start
    output = "".join(content)
    response_path.write_text(output)
    return {
        "wall_s": wall,
        "ttft_s": first,
        "post_first_generation_s": None if first is None else max(0.0, wall - first),
        "usage": usage,
        "finish_reason": finish_reason,
        "output_chars": len(output),
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "reasoning_chars": len("".join(reasoning)),
    }


def wave(base: str, root: Path, articles: list[dict], concurrency: int,
         repeat: int, *, warmup: bool = False) -> dict:
    if profile.http_json(base, "/reset_prefix_cache", {}) != {"success": True}:
        raise RuntimeError("prefix-cache reset failed")
    profile.wait_idle(base)
    label = "warmup-c4" if warmup else f"c{concurrency}-r{repeat + 1}"
    case_dir = root / label
    case_dir.mkdir()
    indices = list(range(4)) if warmup else article_indices(concurrency, repeat)
    before, before_raw = profile.metric_snapshot(base)
    (case_dir / "metrics-before.prom").write_text(before_raw)
    stop = threading.Event()
    samples: list[dict] = []
    monitor_start = time.monotonic()

    def monitor() -> None:
        while not stop.is_set():
            try:
                values, _ = profile.metric_snapshot(base)
                samples.append({"elapsed_s": time.monotonic() - monitor_start, **values})
            except Exception as error:
                samples.append({"error": repr(error)})
            stop.wait(0.25)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    barrier = threading.Barrier(len(indices) + 1)
    started = time.monotonic()
    payloads = []
    results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices)) as executor:
            futures = []
            for position, index in enumerate(indices):
                article = articles[index]
                seed = request_seed(concurrency, repeat, index, position)
                payload = {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": article["prompt"]}],
                    "chat_template_kwargs": {"enable_thinking": False},
                    **SAMPLING,
                    "seed": seed,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                payloads.append({
                    "title": article["title"], "revision_id": article["revision_id"],
                    "prompt_sha256": article["prompt_sha256"], "seed": seed,
                    "sampling": SAMPLING,
                })
                futures.append(executor.submit(
                    stream_chat, base, payload, barrier,
                    case_dir / f"stream-{position + 1}.jsonl",
                    case_dir / f"response-{position + 1}.txt",
                ))
            barrier.wait()
            results = [future.result() for future in futures]
    finally:
        stop.set()
        monitor_thread.join(timeout=5)
    batch_wall = time.monotonic() - started
    profile.wait_idle(base)
    after, after_raw = profile.metric_snapshot(base)
    (case_dir / "metrics-after.prom").write_text(after_raw)
    with (case_dir / "metrics.jsonl").open("w") as output:
        for sample in samples:
            output.write(json.dumps(sample) + "\n")
    (case_dir / "requests.json").write_text(json.dumps(payloads, indent=2) + "\n")
    valid = [sample for sample in samples if "error" not in sample]
    firsts = [row["ttft_s"] for row in results if row["ttft_s"] is not None]
    ends = [row["wall_s"] for row in results]
    overlap_start = max(firsts, default=0)
    overlap_end = min(ends, default=0)
    overlap = [sample for sample in valid if overlap_start <= sample["elapsed_s"] <= overlap_end]
    overlap_tps = None
    if len(overlap) >= 2:
        seconds = overlap[-1]["elapsed_s"] - overlap[0]["elapsed_s"]
        tokens = overlap[-1]["generation_tokens"] - overlap[0]["generation_tokens"]
        if seconds > 0:
            overlap_tps = tokens / seconds

    def change(key: str) -> float:
        return profile.delta(after, before, key)

    completion = sum(row["usage"]["completion_tokens"] for row in results)
    requested_prompt = sum(row["usage"]["prompt_tokens"] for row in results)
    decode_s = change("decode_seconds")
    prefill_s = change("prefill_seconds")
    drafted = change("draft_tokens")
    accepted = change("accepted_tokens")
    row = {
        "label": label, "concurrency": concurrency, "repeat": repeat + 1,
        "warmup": warmup, "requests": [{**meta, **result} for meta, result in zip(payloads, results)],
        "batch_wall_s": batch_wall,
        "completion_tokens": completion,
        "prompt_tokens": requested_prompt,
        "native_decode_tokens_per_s": (completion - len(results)) / decode_s if decode_s > 0 else None,
        "native_prefill_compute_tokens_per_s": change("prefill_computed") / prefill_s if prefill_s > 0 else None,
        "fully_overlapped_aggregate_decode_tokens_per_s": overlap_tps,
        "speculative_draft_tokens": drafted,
        "speculative_accepted_tokens": accepted,
        "speculative_acceptance": accepted / drafted if drafted > 0 else None,
        "prompt_tokens_cached": change("prompt_cached"),
        "preemptions": change("preemptions"),
        "finished_requests": change("finished_requests"),
        "peak_running": max((sample["running"] for sample in valid), default=0),
        "peak_waiting": max((sample["waiting"] for sample in valid), default=0),
        "request_wall_mean_s": statistics.mean(ends),
    }
    if (
        row["finished_requests"] != len(results)
        or change("prompt_tokens") != requested_prompt
        or row["preemptions"] != 0
        or row["prompt_tokens_cached"] != 0
        or drafted <= 0
        or accepted <= 0
        or any(
            result["output_chars"] == 0
            or result["reasoning_chars"] != 0
            or result["usage"]["completion_tokens"] < 128
            or result["finish_reason"] not in ("stop", "length")
            for result in results
        )
    ):
        raise RuntimeError(f"invalid measured wave {label}: {row}")
    (case_dir / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
    print(
        f"DONE {label}: wall={batch_wall:.2f}s, decode={row['native_decode_tokens_per_s']:.2f} tok/s, "
        f"accept={row['speculative_acceptance']:.3f}, tokens={completion}",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    articles = json.loads(args.prompts.read_text())
    if len(articles) != 4:
        raise RuntimeError("expected four frozen articles")
    args.output.mkdir(parents=True, exist_ok=False)
    print("WARMUP C4", flush=True)
    wave(args.base, args.output, articles, 4, 0, warmup=True)
    rows = []
    for concurrency in (1, 2, 3, 4):
        repeats = 4 if concurrency == 1 else 2
        for repeat in range(repeats):
            print(f"START C{concurrency} repeat {repeat + 1}/{repeats}", flush=True)
            rows.append(wave(args.base, args.output, articles, concurrency, repeat))
    (args.output / "results.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()

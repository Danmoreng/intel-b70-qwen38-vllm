#!/usr/bin/env python3
"""Measure native vLLM prefill and decode throughput across context sizes."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
import time
import urllib.request
import uuid
from pathlib import Path


METRICS = (
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_decode_time_seconds_sum",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
)


def post_json(url: str, payload: dict, timeout: int = 60):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def metric_snapshot(root: str) -> dict[str, float]:
    with urllib.request.urlopen(root + "/metrics", timeout=30) as response:
        raw = response.read().decode()
    values: dict[str, float] = {}
    for line in raw.splitlines():
        match = re.match(r"([^#{ ]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if match and match.group(1) in METRICS:
            name = match.group(1)
            values[name] = values.get(name, 0.0) + float(match.group(2))
    return values


def token_count(root: str, model: str, messages: list[dict]) -> int:
    result = post_json(root + "/tokenize", {"model": model, "messages": messages})
    return int(result["count"])


def make_messages(root: str, model: str, target: int, nonce: str) -> list[dict]:
    prefix = (
        f"Run ID {nonce}. Synthetic throughput benchmark. Treat the following "
        "repeated word as opaque context. Reply with a short summary.\n"
    )
    base = [{"role": "user", "content": prefix}]
    base_count = token_count(root, model, base)
    if target <= base_count + 1:
        raise ValueError(f"Context {target} is too small; minimum is {base_count + 2}")
    # After the first occurrence, ` benchmark` is one token with this pinned
    # tokenizer. Verify instead of assuming so the script fails safely on a
    # different model/tokenizer.
    probe1 = [{"role": "user", "content": prefix + " benchmark"}]
    probe2 = [{"role": "user", "content": prefix + " benchmark benchmark"}]
    count1 = token_count(root, model, probe1)
    count2 = token_count(root, model, probe2)
    if count2 - count1 != 1:
        raise RuntimeError("Tokenizer changed: repeated benchmark token is not linear")
    repeats = target - count1 + 1
    messages = [{"role": "user", "content": prefix + " benchmark" * repeats}]
    actual = token_count(root, model, messages)
    if actual != target:
        raise RuntimeError(f"Could not construct exact context: requested={target}, actual={actual}")
    return messages


def run_request(root: str, payload: dict, timeout: int) -> tuple[dict, float, float | None]:
    request = urllib.request.Request(
        root + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first_token = None
    usage = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if first_token is None and (
                    delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                ):
                    first_token = time.monotonic()
    finished = time.monotonic()
    if usage is None:
        raise RuntimeError("Streaming response did not contain usage")
    return usage, finished - started, None if first_token is None else first_token - started


def write_outputs(output_dir: Path, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = [
        "target_context", "prompt_tokens", "completion_tokens", "repeat",
        "prefill_seconds", "prefill_tokens_per_second", "decode_seconds",
        "decode_tokens_per_second", "ttft_seconds", "wall_seconds",
        "accepted_per_drafted", "prefix_cache_hits",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: row.get(key) for key in fields} for row in result["rows"])
    (output_dir / "results.csv").write_text(buffer.getvalue())

    lines = [
        "# Benchmark results",
        "",
        "Medians; native vLLM phase counters; synthetic prompt; prefix-cache misses; "
        "forced fixed-length output. These are throughput measurements, not a quality evaluation.",
        "",
        "| Context | n | Prefill tok/s | Decode tok/s | TTFT | Accepted/drafted |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for context in result["contexts"]:
        rows = [row for row in result["rows"] if row["target_context"] == context]
        if not rows:
            continue
        med = lambda key: statistics.median(row[key] for row in rows)
        lines.append(
            f"| {context:,} | {len(rows)} | {med('prefill_tokens_per_second'):.2f} | "
            f"{med('decode_tokens_per_second'):.2f} | {med('ttft_seconds'):.3f} s | "
            f"{med('accepted_per_drafted'):.1%} |"
        )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="Qwen3.8-27B")
    parser.add_argument("--contexts", default="8192,65536,122880")
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results"))
    args = parser.parse_args()
    contexts = [int(value) for value in args.contexts.split(",")]

    before_all = metric_snapshot(args.root)
    if before_all.get("vllm:num_requests_running", 0) or before_all.get("vllm:num_requests_waiting", 0):
        raise RuntimeError("Server is busy; run this benchmark on an idle single-user server")

    result = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root": args.root,
        "model": args.model,
        "contexts": contexts,
        "output_tokens": args.output_tokens,
        "repeats": args.repeats,
        "rows": [],
    }
    for context in contexts:
        for repeat in range(args.repeats):
            nonce = uuid.uuid4().hex
            messages = make_messages(args.root, args.model, context, nonce)
            payload = {
                "model": args.model,
                "messages": messages,
                "max_tokens": args.output_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            before = metric_snapshot(args.root)
            usage, wall, ttft = run_request(args.root, payload, args.timeout)
            after = metric_snapshot(args.root)
            delta = {name: after.get(name, 0.0) - before.get(name, 0.0) for name in METRICS}
            prefill_s = delta["vllm:request_prefill_time_seconds_sum"]
            decode_s = delta["vllm:request_decode_time_seconds_sum"]
            drafted = delta["vllm:spec_decode_num_draft_tokens_total"]
            accepted = delta["vllm:spec_decode_num_accepted_tokens_total"]
            cache_hits = int(delta["vllm:prefix_cache_hits_total"])
            if usage["prompt_tokens"] != context:
                raise RuntimeError(f"Prompt usage mismatch: {usage}")
            if cache_hits:
                raise RuntimeError(
                    f"Expected a cold unique prompt, but vLLM reported {cache_hits} cache hits"
                )
            row = {
                "target_context": context,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "repeat": repeat,
                "prefill_seconds": prefill_s,
                "prefill_tokens_per_second": context / prefill_s,
                "decode_seconds": decode_s,
                "decode_tokens_per_second": (usage["completion_tokens"] - 1) / decode_s,
                "ttft_seconds": ttft,
                "wall_seconds": wall,
                "accepted_per_drafted": accepted / drafted if drafted else 0.0,
                "prefix_cache_hits": cache_hits,
            }
            result["rows"].append(row)
            write_outputs(args.output_dir, result)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

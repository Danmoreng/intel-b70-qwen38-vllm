#!/usr/bin/env python3
"""Replay one frozen serving request with the original sampling settings."""

import argparse
import hashlib
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path


FROZEN = (Path(__file__).resolve().parents[4] / "intel-b70-qwen38-vllm"
          / "benchmark-results/meaningful-full-profile/run-20260923-201101-w0.00")
METRICS = {
    "prefill_tokens": "vllm:request_prefill_kv_computed_tokens_sum",
    "prefill_s": "vllm:request_prefill_time_seconds_sum",
    "decode_s": "vllm:request_decode_time_seconds_sum",
    "cached_tokens": "vllm:prompt_tokens_cached_total",
    "preemptions": "vllm:num_preemptions_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}


def snapshot(base):
    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()
    values = {key: 0.0 for key in METRICS}
    for line in lines:
        match = re.match(r"([^\s{]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if match:
            for key, name in METRICS.items():
                if match.group(1) == name:
                    values[key] += float(match.group(2))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--case", default="phase-32k-c1")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--arm", default="", help="label the active server flags")
    parser.add_argument("--allow-cache-hits", action="store_true")
    parser.add_argument("--forced-token-id", type=int,
                        help="restrict generation to one token for fixed-continuation timing")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = FROZEN / f"{args.case}-r{args.repeat}"
    frozen = json.loads((folder / "request-1.json").read_text())
    prompt = (folder / "prompt-1.txt").read_text()
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    if digest != frozen["prompt_sha256"]:
        raise RuntimeError("frozen prompt hash mismatch")
    payload = {key: frozen[key] for key in (
        "model", "max_tokens", "temperature", "top_p", "top_k", "seed",
        "ignore_eos", "chat_template_kwargs", "stream", "stream_options")}
    payload["messages"] = [{"role": "user", "content": prompt}]
    if args.forced_token_id is not None:
        payload["allowed_token_ids"] = [args.forced_token_id]
        payload["temperature"] = 0
    before = snapshot(args.base)
    request = urllib.request.Request(args.base + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    start = time.monotonic()
    first = None
    pieces = []
    usage = None
    finish_reason = None
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                part = delta.get("content") or delta.get("reasoning_content") or ""
                pieces.append(part)
                finish_reason = choice.get("finish_reason") or finish_reason
                if first is None and part:
                    first = time.monotonic()
    end = time.monotonic()
    after = snapshot(args.base)
    counts = {key: after[key] - before[key] for key in METRICS}
    image = "local/b70-qwen38-vllm:onednn-poc-20260929"
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        text=True,
    ).strip()
    result = {
        "case": args.case, "repeat": args.repeat, "prompt_sha256": digest,
        "arm": args.arm,
        "forced_token_id": args.forced_token_id,
        "image": image, "image_id": image_id,
        "usage": usage, "finish_reason": finish_reason,
        "output_sha256": hashlib.sha256("".join(pieces).encode()).hexdigest(),
        "ttft_s": None if first is None else first - start,
        "wall_s": end - start, "metrics": counts,
        "native_prefill_tps": (counts["prefill_tokens"] / counts["prefill_s"]
                               if counts["prefill_s"] > 0 else None),
    }
    if (usage or {}).get("completion_tokens") != frozen["max_tokens"]:
        raise RuntimeError("output-token count differs from frozen request")
    if counts["preemptions"] or (counts["cached_tokens"] and not args.allow_cache_hits):
        raise RuntimeError("cold request had preemption or cache hit")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

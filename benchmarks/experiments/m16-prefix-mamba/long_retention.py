#!/usr/bin/env python3
"""One cold 160K prefix, warm 190K extension, then cold 190K replay."""

import argparse
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request


MODEL = "Qwen3.8-27B"
METRICS = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_usage": "vllm:kv_cache_usage_perc",
    "cached": "vllm:prompt_tokens_cached_total",
    "computed": "vllm:request_prefill_kv_computed_tokens_sum",
    "preemptions": "vllm:num_preemptions_total",
    "drafted": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
}


def post(base, path, body, timeout=120):
    request = urllib.request.Request(
        base + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def tokens(base, prompt):
    with post(base, "/tokenize", {"model": MODEL, "prompt": prompt}, timeout=180) as response:
        return json.load(response)["tokens"]


def metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=20) as response:
        lines = response.read().decode().splitlines()
    result = {}
    for key, name in METRICS.items():
        result[key] = sum(
            float(line.split()[-1]) for line in lines
            if line.startswith(name + "{") or line.startswith(name + " ")
        )
    return result


def fit(base, head, word, target):
    count = max(1, target - len(tokens(base, head)))
    for _ in range(12):
        prompt = head + (" " + word) * count
        actual = len(tokens(base, prompt))
        if actual == target:
            return prompt
        count += target - actual
    raise RuntimeError(f"failed to fit {target} tokens; last count {actual}")


def complete(base, prompt, label):
    before = metrics(base)
    if before["running"] or before["waiting"]:
        raise RuntimeError(f"engine busy before {label}: {before}")
    stop = threading.Event()
    samples = []

    def monitor():
        while not stop.wait(0.5):
            try:
                samples.append(metrics(base))
            except Exception as error:
                samples.append({"error": str(error)})

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    request = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps({
            "model": MODEL, "prompt": prompt, "max_tokens": 32,
            "temperature": 0, "seed": 38, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    first = None
    usage = None
    pieces = []
    try:
        with urllib.request.urlopen(request, timeout=1200) as response:
            for line in response:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                event = json.loads(line[6:])
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    text = choice.get("text") or ""
                    if text:
                        if first is None:
                            first = time.monotonic() - start
                        pieces.append(text)
    finally:
        stop.set()
        thread.join(timeout=5)
    wall = time.monotonic() - start
    after = metrics(base)
    if first is None or usage is None:
        raise RuntimeError(f"missing output or usage for {label}")
    output = "".join(pieces)
    valid_samples = [row for row in samples if "error" not in row]
    row = {
        "label": label,
        "usage": usage,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "output_chars": len(output),
        "ttft_s": first,
        "wall_s": wall,
        "delta": {key: after[key] - before[key] for key in METRICS},
        "peak_kv_usage": max((sample["kv_usage"] for sample in valid_samples), default=None),
        "peak_waiting": max((sample["waiting"] for sample in valid_samples), default=None),
        "sample_count": len(valid_samples),
    }
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("warm", "replay"))
    parser.add_argument("--prefix-tokens", type=int, default=160000)
    parser.add_argument("--full-tokens", type=int, default=190000)
    args = parser.parse_args()
    if not 150000 <= args.prefix_tokens < args.full_tokens <= 190000:
        raise ValueError("expected a 150K–190K progressive case")
    head = "B70 long Mamba retention, seed 38. Code at the start: RIVER-731. Reference:"
    prefix = fit(args.base, head, "alpha", args.prefix_tokens)
    full = fit(args.base, prefix, "beta", args.full_tokens)
    prefix_ids, full_ids = tokens(args.base, prefix), tokens(args.base, full)
    if len(prefix_ids) != args.prefix_tokens or len(full_ids) != args.full_tokens:
        raise RuntimeError("incorrect token count")
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise RuntimeError("token prefix is not identical")
    metadata = {
        "prefix_tokens": len(prefix_ids), "full_tokens": len(full_ids),
        "prefix_token_ids_sha256": hashlib.sha256(json.dumps(prefix_ids).encode()).hexdigest(),
        "full_token_ids_sha256": hashlib.sha256(json.dumps(full_ids).encode()).hexdigest(),
    }
    del prefix_ids, full_ids
    print(json.dumps({"prepared": metadata}), flush=True)
    if args.phase == "warm":
        cold_prefix = complete(args.base, prefix, "cold-prefix")
        warm_extension = complete(args.base, full, "warm-extension")
        result = {"metadata": metadata, "rows": [cold_prefix, warm_extension]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    else:
        result = json.loads(args.output.read_text())
        if result["metadata"] != metadata:
            raise RuntimeError("replay prompt differs from warm phase")
        cold_full = complete(args.base, full, "cold-full-replay")
        result["rows"].append(cold_full)
        result["warm_cold_output_equal"] = (
            result["rows"][1]["output_sha256"] == cold_full["output_sha256"]
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

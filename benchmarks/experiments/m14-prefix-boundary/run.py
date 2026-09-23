#!/usr/bin/env python3
"""Short MTP/Mamba prefix-cache check around an attention block boundary."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


METRICS = {
    "hits": "vllm:prefix_cache_hits_total",
    "queries": "vllm:prefix_cache_queries_total",
    "cached": "vllm:prompt_tokens_cached_total",
    "computed": "vllm:request_prefill_kv_computed_tokens_sum",
    "preemptions": "vllm:num_preemptions_total",
    "drafted": "vllm:spec_decode_num_draft_tokens_total",
}


def post(base, path, body, *, timeout=120):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def tokenize(base, prompt):
    with post(base, "/tokenize", {"model": "Qwen3.8-27B", "prompt": prompt}) as response:
        result = json.load(response)
    return result["tokens"]


def make_prompt(base, target, label):
    head = f"B70 prefix boundary {label}. Continue this sequence:"
    count = len(tokenize(base, head))
    words = max(1, target - count)
    for _ in range(12):
        prompt = head + " alpha" * words
        delta = target - len(tokenize(base, prompt))
        if delta == 0:
            return prompt
        words += delta
        if words < 1:
            break
    raise RuntimeError(f"could not construct {target}-token prompt")


def metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=30) as response:
        lines = response.read().decode().splitlines()
    values = {}
    for key, name in METRICS.items():
        selected = [float(line.split()[-1]) for line in lines
                    if line.startswith(name + "{") or line.startswith(name + " ")]
        if not selected:
            raise RuntimeError(f"missing metric {name}")
        values[key] = sum(selected)
    return values


def completion(base, prompt):
    before = metrics(base)
    request = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps({
            "model": "Qwen3.8-27B", "prompt": prompt, "max_tokens": 16,
            "temperature": 0, "seed": 38, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    first = None
    usage = None
    with urllib.request.urlopen(request, timeout=120) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            if first is None and any(choice.get("text") for choice in event.get("choices", [])):
                first = time.monotonic() - start
    after = metrics(base)
    if first is None or usage is None:
        raise RuntimeError("completion produced no streamed token or usage")
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "ttft_s": round(first, 4),
        "wall_s": round(time.monotonic() - start, 4),
        "delta": {key: after[key] - before[key] for key in METRICS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--block-size", type=int, default=1664)
    parser.add_argument("--multiples", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for multiple in args.multiples:
        for offset in (-1, 0, 1):
            target = multiple * args.block_size + offset
            prompt = make_prompt(args.base, target, f"{args.label}-{multiple}-{offset}")
            tokens = tokenize(args.base, prompt)
            extension = prompt + " beta" * 64
            extension_tokens = tokenize(args.base, extension)
            if len(tokens) != target or extension_tokens[:target] != tokens:
                raise RuntimeError(f"token prefix mismatch at {target}")
            row = {
                "target_tokens": target,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "token_ids_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                "extension_tokens": len(extension_tokens),
                "cold": completion(args.base, prompt),
                "resend": completion(args.base, prompt),
                "extension": completion(args.base, extension),
            }
            if any(row[step]["delta"]["preemptions"] != 0 for step in
                   ("cold", "resend", "extension")):
                raise RuntimeError(f"preemption at {target}")
            if row["cold"]["delta"]["hits"] != 0:
                raise RuntimeError(f"cold prompt unexpectedly hit cache at {target}")
            rows.append(row)
            print(json.dumps({"tokens": target,
                              "cache_hits": [row[step]["delta"]["hits"] for step in
                                             ("cold", "resend", "extension")],
                              "ttft_s": [row[step]["ttft_s"] for step in
                                         ("cold", "resend", "extension")]}), flush=True)
    result = {
        "image_label": args.label,
        "block_size": args.block_size,
        "all_warm_hit": all(row[step]["delta"]["hits"] > 0 for row in rows
                            for step in ("resend", "extension")),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare fixed-seed MTP answers with and without prefix caching."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


BASE = "http://127.0.0.1:8081"
INDICES = (17, 211, 399, 577, 799)


def code(index):
    return hashlib.sha256(f"B70-EAGLE-20260923-{index}".encode()).hexdigest()[:12].upper()


def metric(name):
    with urllib.request.urlopen(BASE + "/metrics", timeout=15) as response:
        lines = response.read().decode().splitlines()
    values = [float(line.split()[-1]) for line in lines
              if line.startswith(name + "{") or line.startswith(name + " ")]
    if not values:
        raise RuntimeError(f"missing metric {name}")
    return sum(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("on", "off"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = "EAGLE/Mamba cache-state check. Answer with the exact code only.\n" + "\n".join(
        f"Record {index:04d}: code {code(index)}; value {index * 17 + 3}."
        for index in range(800)
    )
    cases = []
    for index in INDICES:
        body = {
            "model": "Qwen3.8-27B",
            "messages": [
                {"role": "system", "content": document},
                {"role": "user", "content": f"Return only the code for record {index:04d}."},
            ],
            "max_tokens": 32,
            "temperature": 0,
            "seed": 38,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        before = {key: metric(name) for key, name in {
            "hits": "vllm:prefix_cache_hits_total",
            "drafted": "vllm:spec_decode_num_draft_tokens_total",
            "accepted": "vllm:spec_decode_num_accepted_tokens_total",
            "preemptions": "vllm:num_preemptions_total",
        }.items()}
        request = urllib.request.Request(
            BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            answer = json.load(response)
        after = {key: metric(name) for key, name in {
            "hits": "vllm:prefix_cache_hits_total",
            "drafted": "vllm:spec_decode_num_draft_tokens_total",
            "accepted": "vllm:spec_decode_num_accepted_tokens_total",
            "preemptions": "vllm:num_preemptions_total",
        }.items()}
        content = answer["choices"][0]["message"].get("content") or ""
        row = {
            "record": index, "expected": code(index), "answer": content.strip(),
            "prompt_tokens": answer["usage"]["prompt_tokens"],
            "completion_tokens": answer["usage"]["completion_tokens"],
            "metric_delta": {key: after[key] - before[key] for key in before},
        }
        cases.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "mode": args.mode,
        "document_sha256": hashlib.sha256(document.encode()).hexdigest(),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if any(row["answer"] != row["expected"] or
           row["metric_delta"]["preemptions"] != 0 or
           row["metric_delta"]["drafted"] <= 0 for row in cases):
        raise RuntimeError("incorrect answer, preemption, or no MTP")
    if args.mode == "on" and any(row["metric_delta"]["hits"] <= 0
                                 for row in cases[1:]):
        raise RuntimeError("cache-on run lacked warm prefix hits")
    if args.mode == "off" and any(row["metric_delta"]["hits"] != 0
                                  for row in cases):
        raise RuntimeError("cache-off run unexpectedly had prefix hits")


if __name__ == "__main__":
    main()

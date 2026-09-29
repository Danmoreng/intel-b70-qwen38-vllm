#!/usr/bin/env python3
"""Teacher-forced NLL on frozen long source-review prompts via vLLM API."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from replay_frozen import snapshot


FROZEN = (Path(__file__).resolve().parents[4] / "intel-b70-qwen38-vllm"
          / "benchmark-results/meaningful-full-profile/run-20260923-201101-w0.00")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", default="phase-32k-c1")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--start-repeat", type=int, default=1)
    parser.add_argument("--include-token-logprobs", action="store_true")
    args = parser.parse_args()
    rows = []
    for repeat in range(args.start_repeat, args.start_repeat + args.repeats):
        source = FROZEN / f"{args.case}-r{repeat}" / "prompt-1.txt"
        prompt = source.read_text()
        payload = {
            "model": "Qwen3.8-27B", "prompt": prompt,
            "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1,
            "return_token_ids": True,
        }
        request = urllib.request.Request(
            args.base + "/v1/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        before = snapshot(args.base)
        with urllib.request.urlopen(request, timeout=900) as response:
            data = json.load(response)
        after = snapshot(args.base)
        choice = data["choices"][0]
        entries = choice.get("prompt_logprobs") or data.get("prompt_logprobs")
        token_ids = choice.get("prompt_token_ids") or data.get("prompt_token_ids")
        if entries is None or token_ids is None:
            raise RuntimeError(f"prompt logprobs missing; response keys={list(data)}, choice keys={list(choice)}")
        values = [entry[str(token)]["logprob"]
                  for token, entry in zip(token_ids, entries, strict=True)
                  if entry is not None]
        row = {
            "case": args.case, "repeat": repeat,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "tokens": len(values), "nll": -sum(values) / len(values),
            "cached_tokens": after["cached_tokens"] - before["cached_tokens"],
            "prefill_tokens": after["prefill_tokens"] - before["prefill_tokens"],
            "preemptions": after["preemptions"] - before["preemptions"],
        }
        if row["cached_tokens"] or row["preemptions"]:
            raise RuntimeError(f"NLL prompt was cached or preempted: {row}")
        if args.include_token_logprobs:
            row["token_logprobs"] = values
        rows.append(row)
        print(json.dumps({key: value for key, value in row.items()
                          if key != "token_logprobs"}), flush=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

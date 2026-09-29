#!/usr/bin/env python3
"""Compare paired token NLL before and after attention-route changes."""

import argparse
import json
import math
from pathlib import Path


def load(path):
    rows = json.loads(path.read_text())
    if len(rows) != 1:
        raise ValueError(f"expected exactly one prompt in {path}")
    row = rows[0]
    if not row.get("token_ids_sha256"):
        raise ValueError(f"token-ID digest missing from {path}")
    if not row.get("token_positions") or not row.get("token_logprobs"):
        raise ValueError(f"per-token logprobs missing from {path}")
    if not (len(row["token_positions"]) == len(row["token_logprobs"]) == row["tokens"]):
        raise ValueError(f"per-token lengths disagree in {path}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cutpoints", default="13312,29952",
                        help="token positions where a route change begins")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control, candidate = load(args.control), load(args.candidate)
    for key in ("case", "repeat", "prompt_sha256", "token_ids_sha256",
                "token_positions", "tokens"):
        if control[key] != candidate[key]:
            raise ValueError(f"unaligned paired NLL data: {key}")
    if control["cached_tokens"] or candidate["cached_tokens"] or \
       control["preemptions"] or candidate["preemptions"]:
        raise ValueError("scored prompt had a cache hit or preemption")
    positions = control["token_positions"]
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError("scored token positions are not contiguous")
    cutpoints = sorted({int(value) for value in args.cutpoints.split(",") if value})
    if any(point <= positions[0] or point > positions[-1] for point in cutpoints):
        raise ValueError("cutpoint outside scored positions")
    deltas = [before - after for before, after in zip(
        control["token_logprobs"], candidate["token_logprobs"], strict=True)]

    def window(start, end):
        values = [deltas[position - positions[0]] for position in range(start, end)]
        return {
            "start": start, "end_exclusive": end, "tokens": len(values),
            "mean_delta_nll": sum(values) / len(values),
            "perplexity_ratio": math.exp(sum(values) / len(values)),
            "positive_delta_fraction": sum(value > 0 for value in values) / len(values),
            "max_abs_delta_nll": max(map(abs, values)),
        }

    bounds = [positions[0], *cutpoints, positions[-1] + 1]
    result = {
        "case": control["case"], "repeat": control["repeat"],
        "prompt_sha256": control["prompt_sha256"],
        "token_ids_sha256": control["token_ids_sha256"],
        "control_nll": control["nll"], "candidate_nll": candidate["nll"],
        "whole": window(bounds[0], bounds[-1]),
        "windows": [window(start, end) for start, end in zip(bounds, bounds[1:])],
        "after_change": [window(point, bounds[-1]) for point in cutpoints],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"whole": result["whole"], "windows": result["windows"],
                      "after_change": result["after_change"]}), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize the paired long-context practical screen without changing scores."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def rows(path):
    data = [json.loads(line) for line in path.read_text().splitlines()]
    ids = [row["id"] for row in data]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate task IDs in {path}")
    return {row["id"]: row for row in data}


def describe(data):
    result = {"count": len(data), "passed": sum(row["score"]["passed"] for row in data)}
    result["by_kind"] = {
        kind: {"count": len(group), "passed": sum(row["score"]["passed"] for row in group)}
        for kind in sorted({row["kind"] for row in data})
        if (group := [row for row in data if row["kind"] == kind])
    }
    for field in ("ttft_s", "wall_s"):
        result[field] = {
            "median": statistics.median(row[field] for row in data if row[field] is not None),
            "sum": sum(row[field] for row in data if row[field] is not None),
        }
    for field in ("prefill_tokens", "prefill_s", "decode_s", "cached_tokens",
                  "preemptions", "draft_tokens", "accepted_tokens"):
        result[field] = sum(row["metrics"][field] for row in data)
    if result["prefill_s"]:
        result["prefill_tps"] = result["prefill_tokens"] / result["prefill_s"]
    if result["draft_tokens"]:
        result["mtp_acceptance"] = result["accepted_tokens"] / result["draft_tokens"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("performance", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference = rows(args.reference)
    performance = rows(args.performance)
    if set(reference) != set(performance):
        raise ValueError("task ID sets differ")
    for key in reference:
        a, b = reference[key], performance[key]
        for field in ("kind", "context_sha256", "prompt_sha256", "image"):
            if a[field] != b[field]:
                raise ValueError(f"{key}: {field} differs")
        if a["metrics"]["preemptions"] or b["metrics"]["preemptions"]:
            raise ValueError(f"{key}: preemption occurred")
    left = list(reference.values())
    right = [performance[row["id"]] for row in left]
    result = {
        "image": left[0]["image"],
        "reference": describe(left), "performance": describe(right),
        "outcome_changes": {
            "improved": [a["id"] for a, b in zip(left, right)
                         if not a["score"]["passed"] and b["score"]["passed"]],
            "regressed": [a["id"] for a, b in zip(left, right)
                          if a["score"]["passed"] and not b["score"]["passed"]],
        },
        "same_output_count": sum(a["output_sha256"] == b["output_sha256"]
                                 for a, b in zip(left, right)),
        "finish_reasons": {"reference": dict(Counter(a["finish_reason"] for a in left)),
                           "performance": dict(Counter(b["finish_reason"] for b in right))},
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()

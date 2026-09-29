#!/usr/bin/env python3
"""Summarize the exact-length late 128K mixed attention comparison."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text())


def main():
    control = load("mixed-route-32k-128k-control.json")
    candidate = load("mixed-route-32k-128k-candidate.json")
    diagnostic = load("mixed-validate-32k-128k-tail.json")
    if len({run["image_id"] for run in (control, candidate, diagnostic)}) != 1:
        raise ValueError("image IDs differ")
    if len({json.dumps(run["prompt_sha256"], sort_keys=True)
            for run in (control, candidate, diagnostic)}) != 1:
        raise ValueError("prompt hashes differ")
    for label in ("short", "long"):
        hashes = {run["requests"][label]["output_sha256"]
                  for run in (control, candidate, diagnostic)}
        counts = {run["requests"][label]["usage"]["completion_tokens"]
                  for run in (control, candidate, diagnostic)}
        if len(hashes) != 1 or counts != {1024}:
            raise ValueError(f"{label} response changed")
    if diagnostic["metrics"]["cached_tokens"] or diagnostic["metrics"]["preemptions"]:
        raise ValueError("diagnostic cached or preempted")

    groups = {"decode": [], "prefill": []}
    for line in (ROOT / "mixed-validate-32k-128k-tail-results.txt").read_text().splitlines():
        for marker, kind in (("B70_ONEDNN_MIXED_DECODE_VALIDATE ", "decode"),
                             ("B70_ONEDNN_MIXED_VALIDATE ", "prefill")):
            if marker in line:
                groups[kind].append(ast.literal_eval(line.split(marker, 1)[1]))
                break
    if len(groups["decode"]) != 8 or len(groups["prefill"]) != 8:
        raise ValueError("expected eight comparisons of each kind")
    for kind, rows in groups.items():
        if not all(row["allclose"] for row in rows):
            raise ValueError(f"{kind} allclose failure")
        if len({tuple(row["signature"]) for row in rows}) != 8:
            raise ValueError(f"duplicate {kind} signature")
    if min(row["signature"][2] for row in groups["prefill"]) < 95000:
        raise ValueError("prefill comparison missed requested tail")
    if max(row["signature"][2] for row in groups["prefill"]) < 128000:
        raise ValueError("prefill comparison did not reach 128K")

    summary = {
        "image_id": diagnostic["image_id"],
        "attention_result_served": "existing full-batch fallback",
        "diagnostic_min_kv": 95000,
        "tolerance": {"rtol": 0.01, "atol": 0.002},
        "decode": {
            "allclose_count": len(groups["decode"]),
            "max_relative_l2": max(row["relative_l2"] for row in groups["decode"]),
            "max_absolute_error": max(row["max_abs"] for row in groups["decode"]),
            "shapes": groups["decode"],
        },
        "prefill": {
            "allclose_count": len(groups["prefill"]),
            "min_kv": min(row["signature"][2] for row in groups["prefill"]),
            "max_kv": max(row["signature"][2] for row in groups["prefill"]),
            "max_relative_l2": max(row["relative_l2"] for row in groups["prefill"]),
            "max_absolute_error": max(row["max_abs"] for row in groups["prefill"]),
            "shapes": groups["prefill"],
        },
        "limit": "One 32K/128K mixed serving trace; eight sampled late-prefill signatures and eight short decode/MTP signatures, not broad task quality.",
    }
    (ROOT / "mixed-tail-validation-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: {field: value for field, value in rows.items()
                            if field != "shapes"}
                      for key, rows in summary.items()
                      if key in ("decode", "prefill")}))


if __name__ == "__main__":
    main()

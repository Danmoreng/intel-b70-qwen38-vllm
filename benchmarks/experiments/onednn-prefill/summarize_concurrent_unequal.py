#!/usr/bin/env python3
"""Validate and compare the paired unequal 32K/128K C2 serving screen."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    reference = read("concurrent-32k-128k-reference.json")
    candidate = read("concurrent-32k-128k-performance-max196608.json")
    for field in ("image_id", "cases", "prompt_sha256", "forced_token_id"):
        if reference[field] != candidate[field]:
            raise ValueError(f"paired C2 {field} differs")
    if reference["cases"] != {"short": "32k", "long": "128k"}:
        raise ValueError("unexpected frozen cases")
    if reference["forced_token_id"] != 264:
        raise ValueError("unexpected fixed continuation")
    for arm in (reference, candidate):
        if arm["start_skew_s"] >= 0.1:
            raise ValueError("C2 requests did not start together")
        if arm["metrics"]["cached_tokens"] or arm["metrics"]["preemptions"]:
            raise ValueError("C2 cache hit or preemption")
        for row in arm["requests"].values():
            if row["usage"]["completion_tokens"] != 1024:
                raise ValueError("C2 request did not emit 1024 tokens")
    for field in ("prefill_tokens", "cached_tokens", "preemptions"):
        if reference["metrics"][field] != candidate["metrics"][field]:
            raise ValueError(f"paired C2 {field} differs")

    compared = {}
    for label in ("short", "long"):
        a = reference["requests"][label]
        b = candidate["requests"][label]
        if a["usage"]["prompt_tokens"] != b["usage"]["prompt_tokens"]:
            raise ValueError(f"{label} prompt token count differs")
        if a["output_sha256"] != b["output_sha256"]:
            raise ValueError(f"{label} fixed-continuation output differs")
        compared[label] = {
            key: {"reference": a[key], "performance": b[key],
                  "delta": b[key] - a[key]}
            for key in ("ttft_s", "wall_s", "piece_gap_p95_s",
                        "piece_gap_p99_s", "piece_gap_max_s")
        }
    routes = {}
    for name, arm in (("reference", "reference"),
                      ("performance", "performance-max196608")):
        lines = (ROOT / f"concurrent-32k-128k-{arm}-routes.txt").read_text().splitlines()
        routes[name] = {
            "onednn_unique_signatures": sum("B70_ONEDNN_PREFILL_DISPATCH" in line
                                            for line in lines),
            "q128_unique_signatures": sum("B70_Q128_DISPATCH" in line
                                          for line in lines),
        }
    result = {
        "image_id": reference["image_id"],
        "same_fixed_output": True,
        "prefill_s": {
            "reference": reference["metrics"]["prefill_s"],
            "performance": candidate["metrics"]["prefill_s"],
        },
        "kv_cache_peak_fraction": {
            "reference": reference["cache_gauge_peaks"]["vllm:kv_cache_usage_perc"],
            "performance": candidate["cache_gauge_peaks"]["vllm:kv_cache_usage_perc"],
        },
        "requests": compared,
        "route_signatures": routes,
        "limitation": "One fresh-start unequal C2 pair. Stream pieces can contain multiple tokens; gap quantiles are bundle gaps. Mixed metadata fallback was not extended to oneDNN.",
    }
    output = ROOT / "concurrent-32k-128k-summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

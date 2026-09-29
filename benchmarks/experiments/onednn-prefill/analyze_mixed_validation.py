#!/usr/bin/env python3
"""Verify mixed-prefill probe outputs against the existing batch fallback."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    trace = read("mixed-trace-4k-32k.json")
    validation = read("mixed-validate-4k-32k.json")
    if trace["prompt_sha256"] != validation["prompt_sha256"]:
        raise ValueError("mixed trace and validation prompts differ")
    for label in ("short", "long"):
        a, b = trace["requests"][label], validation["requests"][label]
        if (a["output_sha256"] != b["output_sha256"] or
                a["usage"]["completion_tokens"] != 1024 or
                b["usage"]["completion_tokens"] != 1024):
            raise ValueError(f"{label}: fixed continuation differs")
    if validation["metrics"]["preemptions"] or validation["metrics"]["cached_tokens"]:
        raise ValueError("mixed validation had preemption or cache hit")

    expected = {
        (row["request_index"], row["query_rows"], row["active_kv"])
        for row in read("mixed-metadata-contract-4k-32k.json")[
            "potential_single_request_onednn_subcalls"]
    }
    rows = []
    for line in (ROOT / "mixed-validate-4k-32k-results.txt").read_text().splitlines():
        marker = "B70_ONEDNN_MIXED_VALIDATE "
        if marker in line:
            rows.append(ast.literal_eval(line.split(marker, 1)[1]))
    observed = {tuple(row["signature"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError(f"validated mixed signatures differ: {observed} != {expected}")
    if not all(row["allclose"] for row in rows):
        raise ValueError("at least one mixed subcall failed allclose")
    result = {
        "diagnostic_image_id": validation["image_id"],
        "comparison": "oneDNN per-request prefill subcall versus the existing full mixed-batch output",
        "server_response_source": "existing fallback; diagnostic oneDNN output was not returned",
        "validated_shapes": rows,
        "max_relative_l2": max(row["relative_l2"] for row in rows),
        "max_absolute_error": max(row["max_abs"] for row in rows),
        "allclose_count": len(rows),
        "limit": "Five shapes from one 4K/32K C2 scenario; decode/MTP subcall numerics and split-route latency are not established.",
    }
    output = ROOT / "mixed-validation-summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"allclose_count": len(rows),
                      "max_relative_l2": result["max_relative_l2"]}), flush=True)


if __name__ == "__main__":
    main()

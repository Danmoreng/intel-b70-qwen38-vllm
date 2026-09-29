#!/usr/bin/env python3
"""Check separate decode/MTP and prefill subcalls in a real mixed batch."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    baseline = read("mixed-trace-4k-32k.json")
    run = read("mixed-decode-validate-4k-32k.json")
    if baseline["prompt_sha256"] != run["prompt_sha256"]:
        raise ValueError("mixed decode validation prompt changed")
    if run["metrics"]["cached_tokens"] or run["metrics"]["preemptions"]:
        raise ValueError("mixed decode validation cached or preempted")
    for label in ("short", "long"):
        a, b = baseline["requests"][label], run["requests"][label]
        if a["output_sha256"] != b["output_sha256"]:
            raise ValueError(f"{label} fixed output changed")
        if (a["usage"]["completion_tokens"] != 1024 or
                b["usage"]["completion_tokens"] != 1024):
            raise ValueError(f"{label} token count changed")
    groups = {"decode": [], "prefill": []}
    for line in (ROOT / "mixed-decode-validate-4k-32k-results.txt").read_text().splitlines():
        for marker, kind in (("B70_ONEDNN_MIXED_DECODE_VALIDATE ", "decode"),
                             ("B70_ONEDNN_MIXED_VALIDATE ", "prefill")):
            if marker in line:
                groups[kind].append(ast.literal_eval(line.split(marker, 1)[1]))
                break
    if len(groups["decode"]) != 7 or len(groups["prefill"]) != 5:
        raise ValueError("missing real mixed subcall checks")
    for kind, rows in groups.items():
        if len({tuple(row["signature"]) for row in rows}) != len(rows):
            raise ValueError(f"duplicate {kind} signature")
        if not all(row["allclose"] for row in rows):
            raise ValueError(f"{kind} subcall failed allclose")
    expected_prefill = {
        (row["request_index"], row["query_rows"], row["active_kv"])
        for row in read("mixed-metadata-contract-4k-32k.json")[
            "potential_single_request_onednn_subcalls"]
    }
    if {tuple(row["signature"]) for row in groups["prefill"]} != expected_prefill:
        raise ValueError("prefill validation shapes changed")
    result = {
        "diagnostic_image_id": run["image_id"],
        "server_response_source": "existing full-batch fallback",
        "decode": {"allclose_count": len(groups["decode"]),
                   "max_relative_l2": max(row["relative_l2"] for row in groups["decode"]),
                   "max_absolute_error": max(row["max_abs"] for row in groups["decode"]),
                   "shapes": groups["decode"]},
        "prefill": {"allclose_count": len(groups["prefill"]),
                    "max_relative_l2": max(row["relative_l2"] for row in groups["prefill"]),
                    "max_absolute_error": max(row["max_abs"] for row in groups["prefill"]),
                    "shapes": groups["prefill"]},
        "limit": "One 4K/32K C2 trace; separate subcalls have not yet replaced the full batch, and repeated service latency remains unmeasured.",
    }
    (ROOT / "mixed-decode-validation-summary.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps({kind: {"allclose_count": row["allclose_count"],
                             "max_relative_l2": row["max_relative_l2"]}
                      for kind, row in result.items()
                      if kind in ("decode", "prefill")}))


if __name__ == "__main__":
    main()

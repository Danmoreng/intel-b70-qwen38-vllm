#!/usr/bin/env python3
"""Validate and summarize the three paired W4A16 32K attention replays."""

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IMAGE = json.loads((ROOT / "candidate_manifest.json").read_text())["final_image"]


def one(name):
    return json.loads((ROOT / name).read_text())


def nll_rows(names):
    return {row["repeat"]: row for name in names for row in one(name)}


def serving_values(rows, field):
    return [row["metrics"][field] for row in rows]


def main():
    control_nll = nll_rows([
        f"nll-32k-w4a16-q128-isolated-r{repeat}.json" for repeat in (1, 2, 3)
    ])
    candidate_nll = nll_rows([
        "nll-32k-w4a16-onednn-performance-r1-r2.json",
        "nll-32k-w4a16-onednn-performance-r3.json",
    ])
    if set(control_nll) != {1, 2, 3} or set(candidate_nll) != {1, 2, 3}:
        raise ValueError("expected exactly three NLL cases per arm")

    control_serving = []
    candidate_serving = []
    paired = []
    for repeat in (1, 2, 3):
        a = one(f"fixed-32k-w4a16-q128-isolated-r{repeat}.json")
        b = one(f"fixed-32k-w4a16-onednn-performance-r{repeat}.json")
        na, nb = control_nll[repeat], candidate_nll[repeat]
        for key in ("prompt_sha256", "token_ids_sha256", "tokens",
                    "first_scored_position", "last_scored_position",
                    "prefill_tokens"):
            if na[key] != nb[key]:
                raise ValueError(f"repeat {repeat}: NLL {key} differs")
        if na["prompt_sha256"] != a["prompt_sha256"] or a["prompt_sha256"] != b["prompt_sha256"]:
            raise ValueError(f"repeat {repeat}: serving/NLL prompt differs")
        if na["cached_tokens"] or nb["cached_tokens"] or na["preemptions"] or nb["preemptions"]:
            raise ValueError(f"repeat {repeat}: NLL cache hit or preemption")
        for field in ("forced_token_id", "output_sha256"):
            if a[field] != b[field]:
                raise ValueError(f"repeat {repeat}: serving {field} differs")
        if a["forced_token_id"] != 264:
            raise ValueError(f"repeat {repeat}: unexpected forced token")
        for row in (a, b):
            if row["usage"]["completion_tokens"] != 1024 or row["metrics"]["cached_tokens"] or row["metrics"]["preemptions"]:
                raise ValueError(f"repeat {repeat}: serving count/cache/preemption differs")
            if row["image"] != "local/b70-qwen38-vllm:onednn-poc-20260929":
                raise ValueError(f"repeat {repeat}: image tag differs")
        if a["usage"]["prompt_tokens"] != b["usage"]["prompt_tokens"] or a["metrics"]["prefill_tokens"] != b["metrics"]["prefill_tokens"]:
            raise ValueError(f"repeat {repeat}: serving prompt/prefill token count differs")

        control_serving.append(a)
        candidate_serving.append(b)
        paired.append({
            "repeat": repeat, "prompt_sha256": a["prompt_sha256"],
            "control_nll": na["nll"], "candidate_nll": nb["nll"],
            "delta_nll": nb["nll"] - na["nll"],
            "control_prefill_s": a["metrics"]["prefill_s"],
            "candidate_prefill_s": b["metrics"]["prefill_s"],
            "control_wall_s": a["wall_s"], "candidate_wall_s": b["wall_s"],
            "control_decode_s": a["metrics"]["decode_s"],
            "candidate_decode_s": b["metrics"]["decode_s"],
        })

    deltas = [row["delta_nll"] for row in paired]
    result = {
        "image_id": IMAGE,
        "arms": {"control": "W4A16 + Q128", "candidate": "W4A16 + oneDNN performance"},
        "cases": paired,
        "mean_delta_nll": statistics.mean(deltas),
        "sampled_frozen_nll_gate": statistics.mean(deltas) <= 0.01 and max(deltas) <= 0.02,
        "median_prefill_s": {
            "control": statistics.median(serving_values(control_serving, "prefill_s")),
            "candidate": statistics.median(serving_values(candidate_serving, "prefill_s")),
        },
        "median_native_prefill_tps": {
            "control": statistics.median(row["native_prefill_tps"] for row in control_serving),
            "candidate": statistics.median(row["native_prefill_tps"] for row in candidate_serving),
        },
        "median_wall_s": {
            "control": statistics.median(row["wall_s"] for row in control_serving),
            "candidate": statistics.median(row["wall_s"] for row in candidate_serving),
        },
        "median_decode_s": {
            "control": statistics.median(serving_values(control_serving, "decode_s")),
            "candidate": statistics.median(serving_values(candidate_serving, "decode_s")),
        },
    }
    output = ROOT / "factorial-32k-w4a16-attention-summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

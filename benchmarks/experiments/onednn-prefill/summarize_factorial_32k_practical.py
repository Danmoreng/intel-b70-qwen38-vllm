#!/usr/bin/env python3
"""Validate the four-arm 32K task screen and report factor contrasts."""

import json
from pathlib import Path

from summarize_performance_tasks import describe, rows


ROOT = Path(__file__).resolve().parent
FILES = {
    "R0_w4a16_q128": "practical-32k-w4a16-q128.jsonl",
    "R1_w4a8_q128": "practical-32k-w4a8-q128.jsonl",
    "R2_w4a16_onednn": "practical-32k-w4a16-onednn-performance.jsonl",
    "R3_w4a8_onednn": "practical-performance.jsonl",
}


def difference(a, b, data):
    """Return b minus a for score and aggregate request timing."""
    left, right = data[a], data[b]
    return {
        "passed": right["passed"] - left["passed"],
        "wall_s": right["wall_s"]["sum"] - left["wall_s"]["sum"],
        "prefill_s": right["prefill_s"] - left["prefill_s"],
        "decode_s": right["decode_s"] - left["decode_s"],
    }


def main():
    arms = {arm: rows(ROOT / name) for arm, name in FILES.items()}
    baseline = arms["R0_w4a16_q128"]
    if len(baseline) != 30 or any(set(data) != set(baseline) for data in arms.values()):
        raise ValueError("expected the same 30 task IDs in all four arms")
    for task_id, control in baseline.items():
        for arm, data in arms.items():
            row = data[task_id]
            for field in ("kind", "context_sha256", "prompt_sha256", "image"):
                if row[field] != control[field]:
                    raise ValueError(f"{task_id}: {arm} {field} differs")
            for field in ("prefill_tokens", "cached_tokens"):
                if row["metrics"][field] != control["metrics"][field]:
                    raise ValueError(f"{task_id}: {arm} {field} differs")
            if row["usage"]["prompt_tokens"] != control["usage"]["prompt_tokens"]:
                raise ValueError(f"{task_id}: {arm} tokenized prompt differs")
            if row["metrics"]["preemptions"] or row["finish_reason"] != "stop":
                raise ValueError(f"{task_id}: {arm} preempted or did not stop")

    described = {arm: describe(list(data.values())) for arm, data in arms.items()}
    for arm, data in arms.items():
        cold = [row for row in data.values() if row["metrics"]["cached_tokens"] == 0]
        warm = [row for row in data.values() if row["metrics"]["cached_tokens"] > 0]
        code = [row for row in data.values() if row["kind"] == "code"]
        if (len(cold), len(warm), len(code)) != (5, 25, 10):
            raise ValueError(f"{arm}: unexpected cold/warm/code task counts")
        described[arm]["cold_prefill_s"] = sum(row["metrics"]["prefill_s"] for row in cold)
        described[arm]["warm_prefill_s"] = sum(row["metrics"]["prefill_s"] for row in warm)
        described[arm]["code_wall_s"] = sum(row["wall_s"] for row in code)

    contrasts = {
        "w4a8_at_q128": difference("R0_w4a16_q128", "R1_w4a8_q128", described),
        "w4a8_at_onednn": difference("R2_w4a16_onednn", "R3_w4a8_onednn", described),
        "onednn_at_w4a16": difference("R0_w4a16_q128", "R2_w4a16_onednn", described),
        "onednn_at_w4a8": difference("R1_w4a8_q128", "R3_w4a8_onednn", described),
    }
    r0, r1, r2, r3 = (described[key] for key in FILES)
    interaction = {
        "prefill_s": r3["prefill_s"] - r2["prefill_s"] - r1["prefill_s"] + r0["prefill_s"],
        "wall_s": r3["wall_s"]["sum"] - r2["wall_s"]["sum"] - r1["wall_s"]["sum"] + r0["wall_s"]["sum"],
    }
    outcomes = {}
    for name, (left, right) in {
        "onednn_at_w4a16": ("R0_w4a16_q128", "R2_w4a16_onednn"),
        "onednn_at_w4a8": ("R1_w4a8_q128", "R3_w4a8_onednn"),
    }.items():
        outcomes[name] = {
            "improved": [task_id for task_id in baseline
                         if not arms[left][task_id]["score"]["passed"]
                         and arms[right][task_id]["score"]["passed"]],
            "regressed": [task_id for task_id in baseline
                          if arms[left][task_id]["score"]["passed"]
                          and not arms[right][task_id]["score"]["passed"]],
            "same_output_count": sum(arms[left][task_id]["output_sha256"] ==
                                     arms[right][task_id]["output_sha256"]
                                     for task_id in baseline),
        }
    result = {
        "image": next(iter(baseline.values()))["image"],
        "arms": described,
        "contrasts": contrasts,
        "interaction": interaction,
        "onednn_outcomes": outcomes,
        "note": "Decode time includes different generated outputs across some arms; it is not an isolated kernel-speed measure.",
    }
    output = ROOT / "factorial-32k-practical-summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

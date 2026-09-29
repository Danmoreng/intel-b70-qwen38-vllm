#!/usr/bin/env python3
"""Verify and summarize two cold mixed 128K coding comparisons."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAIRS = (
    ("context-128k-1-code-1", "mixed-practical-128k-code"),
    ("context-128k-2-code-1", "mixed-practical-128k-context2-code"),
)


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    pairs = []
    image_ids = set()
    for task_id, stem in PAIRS:
        control = read(stem + "-control.json")
        candidate = read(stem + "-candidate.json")
        if any(row["task_id"] != task_id for row in (control, candidate)):
            raise ValueError("task ID mismatch")
        for key in ("image_id", "task_prompt_sha256", "task_context_sha256",
                    "background_prompt_sha256", "background_forced_token_id"):
            if control[key] != candidate[key]:
                raise ValueError(f"{task_id}: {key} differs")
        image_ids.add(control["image_id"])
        for row in (control, candidate):
            if (row["metrics"]["cached_tokens"] or
                    row["metrics"]["preemptions"] or
                    row["background"]["usage"]["completion_tokens"] != 1024 or
                    row["task_timing"]["finish_reason"] != "stop" or
                    not row["task_score"]["passed"] or
                    len(row["task_score"]["test_results"]) != 3 or
                    not all(test["passed"] for test in
                            row["task_score"]["test_results"])):
                raise ValueError(f"{task_id}: incomplete or failed run")
        if control["background"]["output_sha256"] != candidate["background"]["output_sha256"]:
            raise ValueError("fixed background output differs")
        pairs.append({
            "task_id": task_id,
            "code_passed": {"off": True, "on": True},
            "output_hashes_equal": control["task_output_sha256"] ==
                                   candidate["task_output_sha256"],
            "task_ttft_s": {"off": control["task_timing"]["ttft_s"],
                            "on": candidate["task_timing"]["ttft_s"]},
            "task_wall_s": {"off": control["task_timing"]["wall_s"],
                            "on": candidate["task_timing"]["wall_s"]},
            "background_max_gap_s": {
                "off": control["background"]["piece_gap_max_s"],
                "on": candidate["background"]["piece_gap_max_s"],
            },
            "background_wall_s": {
                "off": control["background"]["wall_s"],
                "on": candidate["background"]["wall_s"],
            },
        })
    if len(image_ids) != 1:
        raise ValueError("different images across contexts")
    result = {
        "image_id": image_ids.pop(),
        "code_tasks_passed": {"off": 2, "on": 2},
        "pairs": pairs,
        "limit": "Two cold 128K coding tasks with a parallel fixed 32K decoder; broader free-generation quality and service distributions remain unmeasured.",
    }
    (ROOT / "mixed-practical-128k-summary.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps({"code_tasks_passed": result["code_tasks_passed"],
                      "image_id": result["image_id"]}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate and aggregate the four-arm seeded Wikipedia MTP benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


ARMS = ("01-candidate", "02-control", "03-control", "04-candidate")
METRICS = (
    "native_decode_tokens_per_s",
    "native_prefill_compute_tokens_per_s",
    "fully_overlapped_aggregate_decode_tokens_per_s",
    "batch_wall_s",
    "speculative_acceptance",
    "completion_tokens",
)


def distribution(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    plan = json.loads((root / "plan.json").read_text())
    articles = json.loads((root / "prompts.json").read_text())
    if [a["prompt_sha256"] for a in articles] != [a["prompt_sha256"] for a in plan["articles"]]:
        raise RuntimeError("frozen prompts differ from plan")
    cases = {}
    runtimes = []
    versions = {}
    dispatch = {}
    for arm in ARMS:
        rows = json.loads((root / arm / "benchmark" / "results.json").read_text())["cases"]
        if len(rows) != 10 or sum(len(row["requests"]) for row in rows) != 22:
            raise RuntimeError(f"{arm}: expected ten waves and 22 requests")
        if [row["concurrency"] for row in rows] != [1] * 4 + [2] * 2 + [3] * 2 + [4] * 2:
            raise RuntimeError(f"{arm}: scenario order changed")
        for row in rows:
            if (
                row["prompt_tokens_cached"] != 0
                or row["preemptions"] != 0
                or row["speculative_draft_tokens"] <= 0
                or row["speculative_accepted_tokens"] <= 0
                or row["finished_requests"] != row["concurrency"]
                or row["native_decode_tokens_per_s"] is None
                or row["fully_overlapped_aggregate_decode_tokens_per_s"] is None
            ):
                raise RuntimeError(f"{arm}/{row['label']}: invalid wave")
            for request in row["requests"]:
                if (
                    request["output_chars"] <= 0
                    or request["reasoning_chars"] != 0
                    or not 128 <= request["usage"]["completion_tokens"] <= plan["sampling"]["max_tokens"]
                    or request["finish_reason"] not in ("stop", "length")
                ):
                    raise RuntimeError(f"{arm}/{row['label']}: invalid response")
        cases[arm] = rows
        runtime = json.loads((root / arm / "runtime.json").read_text())
        expected_image = plan["candidate_image" if "candidate" in arm else "control_image"]
        if runtime["image"] != expected_image or "--speculative-config" not in runtime["args"]:
            raise RuntimeError(f"{arm}: unexpected image or MTP configuration")
        runtimes.append(runtime)
        versions[arm] = (root / arm / "versions.txt").read_text().splitlines()
        expected_version = "0.30.0+xpu" if "candidate" in arm else "0.29.0+xpu"
        if versions[arm] != [expected_version, "0.1.14.1"]:
            raise RuntimeError(f"{arm}: unexpected versions")
        log = (root / arm / "server.log").read_text()
        dispatch[arm] = {
            "q128": "B70_Q128_DISPATCH" in log,
            "m04": "B70_M04_SHARED_KV_DISPATCH" in log,
            "v2_runner": "Using V2 Model Runner" in log,
        }
    if any(
        runtime["args"] != runtimes[0]["args"]
        or runtime["environment"] != runtimes[0]["environment"]
        for runtime in runtimes[1:]
    ):
        raise RuntimeError("serving arguments or extra environment differ")

    request_keys = {}
    output_hashes = {}
    for arm, rows in cases.items():
        request_keys[arm] = [
            (row["label"], position, request["prompt_sha256"], request["seed"])
            for row in rows
            for position, request in enumerate(row["requests"])
        ]
        output_hashes[arm] = [request["output_sha256"] for row in rows for request in row["requests"]]
    if any(request_keys[arm] != request_keys[ARMS[0]] for arm in ARMS[1:]):
        raise RuntimeError("prompts, seeds or request order differ between arms")

    scenarios = {}
    for concurrency in (1, 2, 3, 4):
        result = {}
        for version in ("candidate", "control"):
            selected = [
                row for arm, rows in cases.items() if version in arm
                for row in rows if row["concurrency"] == concurrency
            ]
            expected = 8 if concurrency == 1 else 4
            if len(selected) != expected:
                raise RuntimeError(f"C{concurrency}/{version}: expected {expected} waves")
            result[version] = {metric: distribution([row[metric] for row in selected]) for metric in METRICS}
            drafted = sum(row["speculative_draft_tokens"] for row in selected)
            accepted = sum(row["speculative_accepted_tokens"] for row in selected)
            result[version]["weighted_speculative_acceptance"] = accepted / drafted
            result[version]["total_drafted_tokens"] = drafted
            result[version]["total_accepted_tokens"] = accepted
        result["candidate_vs_control_pct"] = {
            metric: 100 * (result["candidate"][metric]["mean"] / result["control"][metric]["mean"] - 1)
            for metric in METRICS
        }
        result["paired_aggregate_decode_change_pct"] = [
            100 * (
                candidate["fully_overlapped_aggregate_decode_tokens_per_s"]
                / control["fully_overlapped_aggregate_decode_tokens_per_s"] - 1
            )
            for candidate_arm, control_arm in (
                ("01-candidate", "02-control"),
                ("04-candidate", "03-control"),
            )
            for candidate, control in zip(
                [row for row in cases[candidate_arm] if row["concurrency"] == concurrency],
                [row for row in cases[control_arm] if row["concurrency"] == concurrency],
            )
        ]
        scenarios[f"c{concurrency}"] = result

    c1_by_article = {}
    for article in articles:
        title = article["title"]
        c1_by_article[title] = {}
        for version in ("candidate", "control"):
            selected = [
                (wave, row) for arm, rows in cases.items() if version in arm
                for wave in rows if wave["concurrency"] == 1
                for row in wave["requests"] if row["title"] == title
            ]
            if len(selected) != 2:
                raise RuntimeError(f"C1 article coverage changed: {title}/{version}")
            c1_by_article[title][version] = {
                "mean_completion_tokens": statistics.mean(row["usage"]["completion_tokens"] for _, row in selected),
                "mean_native_decode_tokens_per_s": statistics.mean(wave["native_decode_tokens_per_s"] for wave, _ in selected),
                "mean_speculative_acceptance": statistics.mean(wave["speculative_acceptance"] for wave, _ in selected),
                "output_hashes_identical_between_repeats": selected[0][1]["output_sha256"] == selected[1][1]["output_sha256"],
            }

    summary = {
        "run_id": root.name,
        "plan": plan,
        "validation": {
            "measured_waves": 40,
            "measured_requests": 88,
            "matching_prompt_hashes_and_seeds_across_arms": True,
            "matching_server_args_and_extra_environment": True,
            "no_prefix_cache_hits_or_preemptions": True,
            "all_outputs_nonempty_without_reasoning": True,
            "versions": versions,
            "dispatch": dispatch,
            "matching_output_hashes_between_control_repeats": sum(
                left == right for left, right in zip(output_hashes["02-control"], output_hashes["03-control"])
            ),
            "matching_output_hashes_between_candidate_repeats": sum(
                left == right for left, right in zip(output_hashes["01-candidate"], output_hashes["04-candidate"])
            ),
            "requests_per_arm": 22,
        },
        "scenarios": scenarios,
        "c1_by_article": c1_by_article,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

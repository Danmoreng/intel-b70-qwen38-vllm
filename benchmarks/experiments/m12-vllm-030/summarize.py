#!/usr/bin/env python3
"""Publish a content-free aggregate of the short vLLM 0.30 A/B screen."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import statistics


ARMS = ("01-candidate", "02-control", "03-control", "04-candidate")
METRICS = (
    "native_prefill_compute_tokens_per_s",
    "native_weighted_decode_tokens_per_s",
    "fully_overlapped_aggregate_decode_tokens_per_s",
    "batch_wall_s",
)


def read_one(pattern: str) -> dict:
    paths = glob.glob(pattern)
    if len(paths) != 1:
        raise RuntimeError(f"expected one file for {pattern}, found {len(paths)}")
    return json.loads(Path(paths[0]).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    plan = json.loads((root / "plan.json").read_text())
    mtp_enabled = plan.get("mtp_enabled", True)
    metrics = METRICS + (("speculative_acceptance",) if mtp_enabled else ())
    cases: dict[str, list[dict]] = {}
    runtimes = []
    versions = {}
    output_hashes = {}
    dispatch = {}
    for arm in ARMS:
        result = read_one(str(root / arm / "benchmark" / "run-*" / "results.json"))
        rows = result["cases"]
        if len(rows) != 4:
            raise RuntimeError(f"{arm}: expected four measured waves")
        for row in rows:
            if not (
                row["all_outputs_nonempty"]
                and row["all_completion_counts_exact"]
                and row["all_finish_reasons_length"]
                and row["preemptions"] == 0
                and row["prompt_tokens_cached"] == 0
            ):
                raise RuntimeError(f"{arm}: failed request validation")
        log = (root / arm / "server.log").read_text()
        dispatch[arm] = {
            "q128": "B70_Q128_DISPATCH" in log,
            "m04": "B70_M04_SHARED_KV_DISPATCH" in log,
        }
        if mtp_enabled and not dispatch[arm]["q128"]:
            raise RuntimeError(f"{arm}: Q128 dispatch missing")
        if mtp_enabled != dispatch[arm]["m04"]:
            raise RuntimeError(f"{arm}: unexpected M04 dispatch state")
        if not mtp_enabled and any(
            row["speculative_draft_tokens"] != 0 or row["speculative_accepted_tokens"] != 0
            for row in rows
        ):
            raise RuntimeError(f"{arm}: speculative tokens observed with MTP disabled")
        cases[arm] = rows
        runtime = json.loads((root / arm / "runtime.json").read_text())
        expected_image = plan["candidate_image" if "candidate" in arm else "control_image"]
        if runtime["image"] != expected_image:
            raise RuntimeError(f"{arm}: image differs from plan")
        runtimes.append(runtime)
        versions[arm] = (root / arm / "versions.txt").read_text().splitlines()
        expected_version = "0.30.0+xpu" if "candidate" in arm else "0.29.0+xpu"
        if versions[arm] != [expected_version, "0.1.14.1"]:
            raise RuntimeError(f"{arm}: unexpected vLLM or XPU kernel version")
        output_hashes[arm] = {
            (path.parent.name, index): request["output_sha256"]
            for path in (root / arm / "benchmark").glob("run-*/v030-*-r*/summary.json")
            for index, request in enumerate(json.loads(path.read_text())["requests"])
        }
        if len(output_hashes[arm]) != 10:
            raise RuntimeError(f"{arm}: expected ten measured output hashes")
    if any(
        item["args"] != runtimes[0]["args"]
        or item["environment"] != runtimes[0]["environment"]
        for item in runtimes[1:]
    ):
        raise RuntimeError("server args or extra environment differ between arms")
    if any(("--speculative-config" in item["args"]) != mtp_enabled for item in runtimes):
        raise RuntimeError("speculative configuration differs from plan")

    prompt_hashes: dict[str, list[str]] = {}
    for arm in ARMS:
        for request_path in (root / arm / "benchmark").glob("run-*/v030-*-r*/request-*.json"):
            key = "/".join(request_path.parts[-2:])
            prompt_hashes.setdefault(key, []).append(
                json.loads(request_path.read_text())["prompt_sha256"]
            )
    if len(prompt_hashes) != 10 or any(
        len(hashes) != len(ARMS) or len(set(hashes)) != 1
        for hashes in prompt_hashes.values()
    ):
        raise RuntimeError("prompt hashes differ or requests are missing")
    if any(set(hashes) != set(output_hashes[ARMS[0]]) for hashes in output_hashes.values()):
        raise RuntimeError("output hash keys differ between arms")

    scenarios: dict[str, dict] = {}
    for scenario in ("v030-4k-c1", "v030-4k-c4"):
        scenario_out: dict[str, dict] = {}
        for version in ("candidate", "control"):
            selected = [
                row for arm, rows in cases.items() if version in arm
                for row in rows if row["scenario"]["name"] == scenario
            ]
            if len(selected) != 4:
                raise RuntimeError(f"{scenario}/{version}: expected four values")
            scenario_out[version] = {
                key: {
                    "mean": statistics.mean(row[key] for row in selected),
                    "min": min(row[key] for row in selected),
                    "max": max(row[key] for row in selected),
                    "values": [row[key] for row in selected],
                }
                for key in metrics
            }
        scenario_out["candidate_vs_control_pct"] = {
            key: 100 * (
                scenario_out["candidate"][key]["mean"]
                / scenario_out["control"][key]["mean"] - 1
            )
            for key in metrics
        }
        scenarios[scenario] = scenario_out
    output = {
        "run_id": root.name,
        "plan": plan,
        "validation": {
            "measured_waves": 16,
            "request_payloads_with_matching_prompt_hashes": len(prompt_hashes),
            "server_args_and_extra_environment_equal": True,
            "custom_dispatch_by_arm": dispatch,
            "all_outputs_nonempty": True,
            "all_output_lengths_exact": True,
            "all_finish_reasons_length": True,
            "all_prefix_cache_hits_zero": True,
            "all_preemptions_zero": True,
            "vllm_and_xpu_kernel_versions": versions,
            "output_hash_matches_between_control_repeats": sum(
                output_hashes["02-control"][key] == output_hashes["03-control"][key]
                for key in output_hashes["02-control"]
            ),
            "measured_requests_per_arm": len(output_hashes[ARMS[0]]),
        },
        "scenarios": scenarios,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

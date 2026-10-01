#!/usr/bin/env python3
"""Grade the unmodified output after a timed coding session, including failure."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "web_runner", REPO / "scripts/run-web-coding-benchmark.py"
)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    provenance = json.loads((root / "provenance.json").read_text())
    candidates = [
        p
        for p in (REPO / "benchmarks/web-coding-fixture").glob("*/manifest.json")
        if R.sha(p) == provenance["fixture_manifest_sha256"]
    ]
    if len(candidates) != 1:
        raise RuntimeError("cannot identify the exact timed fixture")
    manifest = json.loads(candidates[0].read_text())
    R.HARNESS = REPO / "benchmarks/web-coding-harness" / manifest["harness_version"]
    if {f: R.sha(R.HARNESS / f) for f in provenance["harness_files"]} != provenance[
        "harness_files"
    ]:
        raise RuntimeError("acceptance harness changed")
    project = root / "project"
    before = R.tree(project)
    started = time.monotonic()
    acceptance = R.run_node("checks.mjs", project, 6, "--acceptance")
    browser = subprocess.run(
        [
            "node",
            str(R.HARNESS / "browser.mjs"),
            str(project),
            "6",
            str(root / "post-run.png"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    try:
        browser_result = json.loads(browser.stdout)
    except ValueError as e:
        raise RuntimeError("browser grading failed: " + browser.stderr[-2000:]) from e
    if acceptance.get("harness_error") or browser_result.get("harness_error"):
        raise RuntimeError("acceptance infrastructure failed")
    if R.tree(project) != before:
        raise RuntimeError("grading changed generated output")
    cases = acceptance["checks"] + [
        x for x in browser_result["checks"] if x["name"] != "browser renderer identity"
    ]
    expected_cases = manifest.get("expected_final_cases", 45)
    if len(cases) != expected_cases:
        raise RuntimeError("acceptance case count differs")
    result = {
        "purpose": "Independent final-output grading after timed session; no model feedback or source repairs; excluded from native and session timing",
        "graded_at": R.now(),
        "wall_s": time.monotonic() - started,
        "fixture_manifest_sha256": provenance["fixture_manifest_sha256"],
        "harness_files": provenance["harness_files"],
        "expected_final_cases": expected_cases,
        "project_sha256": hashlib.sha256(
            json.dumps(before, sort_keys=True).encode()
        ).hexdigest(),
        "acceptance": acceptance,
        "browser": browser_result,
        "passed": sum(x["passed"] for x in cases),
        "total": len(cases),
    }
    R.save(root / "post-run-acceptance.json", result)
    print(
        json.dumps(
            {k: result[k] for k in ["passed", "total", "wall_s", "project_sha256"]}
        )
    )


if __name__ == "__main__":
    main()

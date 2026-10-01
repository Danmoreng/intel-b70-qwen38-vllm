#!/usr/bin/env python3
"""Review the unchanged timed outputs with the corrected storage-access rule."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "runner", REPO / "scripts/run-web-coding-benchmark.py"
)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    project = root / "project"
    original = json.loads((root / "post-run-acceptance.json").read_text())
    before = R.tree(project)
    digest = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    if digest != original["project_sha256"]:
        raise RuntimeError("generated project changed")
    R.HARNESS = REPO / "benchmarks/web-coding-harness/v4"
    if R.sha(R.HARNESS / "browser.mjs") != original["harness_files"]["browser.mjs"]:
        raise RuntimeError("browser cases differ")
    node = R.run_node("checks.mjs", project, 6, "--acceptance")
    R.HARNESS = REPO / "benchmarks/web-coding-harness"
    probe = subprocess.run(
        R.isolated_node(
            project, ["/harness/review-scores.mjs", "/project"], readonly=True
        ),
        capture_output=True,
        text=True,
        timeout=75,
        check=False,
    )
    storage = json.loads(probe.stdout)
    if R.tree(project) != before or node.get("harness_error"):
        raise RuntimeError("review infrastructure failed or changed source")
    cases = node["checks"] + [
        x
        for x in original["browser"]["checks"]
        if x["name"] != "browser renderer identity"
    ]
    assert len(cases) == 54
    result = {
        "purpose": "Post-run contract review, not feedback to the timed agent. Replaces only the overly strict storage-error case with a construction-or-access check; unchanged browser cases reused by identical harness hash.",
        "reviewed_at": R.now(),
        "project_sha256": digest,
        "original_score": {"passed": original["passed"], "total": original["total"]},
        "corrected_checks_sha256": R.sha(
            REPO / "benchmarks/web-coding-harness/v4/checks.mjs"
        ),
        "original_browser_sha256": original["harness_files"]["browser.mjs"],
        "acceptance": node,
        "browser": original["browser"],
        "supplemental_storage": storage,
        "passed": sum(x["passed"] for x in cases),
        "total": len(cases),
    }
    R.save(root / "post-run-contract-review.json", result)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "total": result["total"],
                "supplemental_storage": storage,
            }
        )
    )


if __name__ == "__main__":
    main()

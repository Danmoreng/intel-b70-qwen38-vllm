#!/usr/bin/env python3
"""Short vLLM 0.29/0.30 serving comparison with production recovery."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
COMMON_PATH = REPO / "benchmarks/experiments/m11-pro-review/common.py"
spec = importlib.util.spec_from_file_location("m11_common", COMMON_PATH)
assert spec and spec.loader
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

CONTROL = common.BASE
CANDIDATE = "local/b70-vllm:030-initial"
CANDIDATE_ID = "sha256:05e0981ea0a37ed82b309dbba3157f57c9a8144aadbf2bcb8746bfed70d95016"


def image_id(name: str) -> str:
    return subprocess.check_output(
        ["docker", "image", "inspect", name, "--format", "{{.Id}}"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if image_id(CONTROL) != CONTROL or image_id(CANDIDATE) != CANDIDATE_ID:
        parser.error("an image ID changed; review and update the frozen manifest")
    run_dir = args.run_dir.resolve()
    plan = {
        "control_image": CONTROL,
        "candidate_image": CANDIDATE_ID,
        "candidate_base_digest": "sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d",
        "order": ["candidate", "control", "control", "candidate"],
        "scenario_file": "scenarios.json",
        "sampling": {"temperature": 0},
        "max_num_batched_tokens": 6656,
        "max_run_minutes_before_recovery": 27,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    deadline = time.monotonic() + 27 * 60
    try:
        with common.Session(run_dir) as session:
            for index, arm in enumerate(plan["order"]):
                if time.monotonic() > deadline - 300:
                    raise TimeoutError("screen budget exhausted before next arm")
                image = CANDIDATE if arm == "candidate" else CONTROL
                label = f"{index + 1:02d}-{arm}"
                out = session.start(label, image, budget=6656, cache_key=arm)
                versions = subprocess.check_output([
                    "docker", "exec", common.NAME, "python", "-c",
                    "import importlib.metadata as m; print(m.version('vllm')); print(m.version('vllm-xpu-kernels'))",
                ], text=True, timeout=30)
                (out / "versions.txt").write_text(versions)
                cmd = [
                    sys.executable, str(REPO / "scripts/current-profile-benchmark.py"),
                    "--execute", "--base", common.diag.EXPERIMENT_URL,
                    "--container", common.NAME,
                    "--scenarios", str(HERE / "scenarios.json"),
                    "--output-root", str(out / "benchmark"),
                ]
                remaining = max(1, int(deadline - time.monotonic()))
                with (out / "benchmark.log").open("w") as output:
                    subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT,
                                   check=True, timeout=remaining)
                log = (out / "server.log").read_text()
                if "B70_Q128_DISPATCH" not in log or "B70_M04_SHARED_KV_DISPATCH" not in log:
                    raise RuntimeError(f"Q128/M04 dispatch missing in {label}")
                session.stop()
    finally:
        common.recover(run_dir)


if __name__ == "__main__":
    main()

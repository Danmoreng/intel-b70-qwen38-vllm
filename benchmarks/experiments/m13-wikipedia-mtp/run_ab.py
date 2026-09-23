#!/usr/bin/env python3
"""Run the frozen Wikipedia MTP workload on vLLM 0.30/0.29/0.29/0.30."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location(
    "m13_common", REPO / "benchmarks/experiments/m11-pro-review/common.py"
)
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
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if image_id(CONTROL) != CONTROL or image_id(CANDIDATE) != CANDIDATE_ID:
        parser.error("an image ID changed; review the frozen manifest")
    source = args.prompts.resolve()
    articles = json.loads(source.read_text())
    if len(articles) != 4 or any(
        len(row["prompt_sha256"]) != 64
        or hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]
        or not 3900 <= row["prompt_tokens"] <= 4096
        for row in articles
    ):
        parser.error("frozen prompts failed validation")
    plan = {
        "control_image": CONTROL,
        "candidate_image": CANDIDATE_ID,
        "order": ["candidate", "control", "control", "candidate"],
        "mtp_enabled": True,
        "sampling": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512},
        "seed_rule": "770000 + concurrency*1000 + repeat*100 + article_index*10 + position",
        "concurrencies": [1, 2, 3, 4],
        "repeats": {"c1": 4, "c2": 2, "c3": 2, "c4": 2},
        "max_run_minutes_before_recovery": 27,
        "articles": [
            {key: row[key] for key in (
                "title", "revision_id", "revision_url", "article_sha256",
                "excerpt_sha256", "prompt_sha256", "prompt_tokens",
            )}
            for row in articles
        ],
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, run_dir / "prompts.json")
    (run_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    deadline = time.monotonic() + 27 * 60
    try:
        with common.Session(run_dir) as session:
            for index, arm in enumerate(plan["order"]):
                if time.monotonic() > deadline - 300:
                    raise TimeoutError("screen budget exhausted before next arm")
                image = CANDIDATE if arm == "candidate" else CONTROL
                label = f"{index + 1:02d}-{arm}"
                out = session.start(label, image, budget=6656, cache_key=arm + "-wiki-mtp")
                versions = subprocess.check_output([
                    "docker", "exec", common.NAME, "python", "-c",
                    "import importlib.metadata as m; print(m.version('vllm')); print(m.version('vllm-xpu-kernels'))",
                ], text=True, timeout=30)
                (out / "versions.txt").write_text(versions)
                cmd = [
                    sys.executable, str(HERE / "measure.py"),
                    "--base", common.diag.EXPERIMENT_URL,
                    "--prompts", str(run_dir / "prompts.json"),
                    "--output", str(out / "benchmark"),
                ]
                remaining = max(1, int(deadline - time.monotonic()))
                with (out / "benchmark.log").open("w") as output:
                    subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT,
                                   check=True, timeout=remaining)
                session.stop()
    finally:
        common.recover(run_dir)


if __name__ == "__main__":
    main()

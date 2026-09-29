#!/usr/bin/env python3
"""Score a cold 128K coding task while a frozen 32K request decodes."""

import argparse
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

from concurrent_unequal import describe_request
from performance_tasks_128k import tasks
from replay_frozen import METRICS, snapshot
from run_performance_tasks import request, score
from staggered_serving import payload, stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--task-id", default="context-128k-1-code-1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a fresh file for each cold server")
    task = next((row for row in tasks() if row["id"] == args.task_id), None)
    if task is None or task["kind"] != "code":
        parser.error("task ID must name a frozen 128K coding task")
    short_body, short_hash = payload("32k")
    short_body["allowed_token_ids"] = [264]
    short_body["temperature"] = 0
    before = snapshot(args.base)
    started = time.monotonic()
    short = {"first_piece": threading.Event()}
    long = {}

    def run_long():
        try:
            long["started_s"] = time.monotonic() - started
            long["output"], long["timing"] = request(args.base, task["prompt"])
            long["finished_s"] = time.monotonic() - started
        except Exception as error:
            long["error"] = repr(error)

    short_thread = threading.Thread(target=stream,
        args=(args.base, short_body, short, started), daemon=True)
    long_thread = threading.Thread(target=run_long, daemon=True)
    short_thread.start()
    long_thread.start()
    short_thread.join(timeout=900)
    long_thread.join(timeout=900)
    if short_thread.is_alive() or long_thread.is_alive():
        raise TimeoutError("mixed practical request did not finish")
    if "error" in short or "error" in long:
        raise RuntimeError(f"short={short.get('error')}; long={long.get('error')}")
    after = snapshot(args.base)
    metrics = {key: after[key] - before[key] for key in METRICS}
    if metrics["cached_tokens"] or metrics["preemptions"]:
        raise RuntimeError("mixed practical run cached or preempted")
    if short["usage"]["completion_tokens"] != 1024:
        raise RuntimeError("background 32K request did not emit 1024 tokens")
    image_id = subprocess.check_output(
        ["docker", "inspect", "b70-qwen38-vllm", "--format", "{{.Image}}"],
        text=True).strip()
    output = long["output"]
    result = {
        "arm": args.arm, "image_id": image_id,
        "task_id": task["id"],
        "task_context_sha256": task["context_sha256"],
        "task_prompt_sha256": hashlib.sha256(task["prompt"].encode()).hexdigest(),
        "background_prompt_sha256": short_hash,
        "background_forced_token_id": 264,
        "task_output": output,
        "task_output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "task_score": score(task, output),
        "task_timing": long["timing"],
        "background": describe_request(short),
        "metrics": metrics,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("task_output", "background")}), flush=True)


if __name__ == "__main__":
    main()

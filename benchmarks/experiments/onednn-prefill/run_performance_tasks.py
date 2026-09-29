#!/usr/bin/env python3
"""Paired task-quality and latency screen for reference/performance profiles."""

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

from performance_tasks import tasks as tasks_32k
from performance_tasks_128k import tasks as tasks_128k
from performance_tasks_199k import tasks as tasks_199k
from replay_frozen import METRICS, snapshot


CHECKER = r'''import json
import pathlib
import resource

resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
data = json.loads(pathlib.Path('/task/input.json').read_text())
namespace = {'__name__': '__submission__'}
exec(compile(data['code'], '<submission>', 'exec'), namespace)
solve = namespace['solve']
results = []
for args, expected in data['cases']:
    actual = solve(*args)
    try:
        actual = json.loads(json.dumps(actual))
    except (TypeError, ValueError):
        pass
    results.append({'passed': actual == expected, 'actual': actual})
print(json.dumps({'passed': all(row['passed'] for row in results),
                  'results': results}))
'''


def extract_json(output):
    output = output.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output, re.DOTALL)
    if fence:
        output = fence.group(1)
    else:
        start, end = output.find("{"), output.rfind("}")
        if start < 0 or end < start:
            raise ValueError("JSON object absent")
        output = output[start:end + 1]
    return json.loads(output)


def extract_code(output):
    fence = re.search(r"```(?:python)?\s*(.*?)```", output, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    candidates.append(output)
    marker = output.find("def solve(")
    if marker >= 0:
        candidates.append(output[marker:])
    for candidate in candidates:
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        if any(isinstance(node, ast.FunctionDef) and node.name == "solve"
               for node in tree.body):
            return candidate
    raise ValueError("valid Python solve function absent")


def check_code(output, cases):
    code = extract_code(output)
    with tempfile.TemporaryDirectory(prefix="b70-task-check-") as temporary:
        root = Path(temporary)
        root.chmod(0o755)
        (root / "input.json").write_text(json.dumps({"code": code, "cases": cases}))
        (root / "check.py").write_text(CHECKER)
        (root / "input.json").chmod(0o644)
        (root / "check.py").chmod(0o644)
        name = "b70-task-check-" + uuid.uuid4().hex[:12]
        command = [
            "docker", "run", "--rm", "--name", name, "--network", "none",
            "--read-only", "--pids-limit", "32", "--memory", "256m",
            "--cpus", "1", "--user", "65534:65534", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "--mount", f"type=bind,source={root},target=/task,readonly",
            "--entrypoint", "python", "local/b70-oneapi-2026.1.1-builder:vllm030",
            "-I", "/task/check.py",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=12, check=False)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True,
                           text=True, timeout=10, check=False)
            return {"passed": False, "error": "sandbox timeout"}
        if result.returncode:
            return {"passed": False, "error": (result.stderr or result.stdout)[-1200:]}
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"passed": False, "error": result.stdout[-1200:]}
        return {"passed": report["passed"], "test_results": report["results"]}


def score(task, output):
    try:
        if task["kind"] == "code":
            return check_code(output, task["cases"])
        if task["kind"] == "retrieval":
            answer = output.strip().strip("` ").splitlines()[0]
            return {"passed": answer == task["expected"], "answer": answer}
        answer = extract_json(output)
        expected = task["expected"]
        if task["kind"] == "review":
            return {"passed": answer == expected, "answer": answer}
        if set(answer) != set(expected):
            return {"passed": False, "answer": answer}
        passed = all(isinstance(answer[key], (int, float)) and
                     abs(answer[key] - value) <= 0.0001
                     for key, value in expected.items())
        return {"passed": passed, "answer": answer}
    except (ValueError, KeyError, IndexError, SyntaxError) as error:
        return {"passed": False, "error": str(error)}


def request(base, prompt):
    payload = {
        "model": "Qwen3.8-27B", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 768, "temperature": 0, "top_p": 1.0, "top_k": 20,
        "seed": 70, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    before = snapshot(base)
    start = time.monotonic()
    first = None
    chunks = []
    usage = None
    finish_reason = None
    req = urllib.request.Request(base + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    if first is None:
                        first = time.monotonic()
                    chunks.append(piece)
                finish_reason = choice.get("finish_reason") or finish_reason
    end = time.monotonic()
    after = snapshot(base)
    output = "".join(chunks)
    metrics = {key: after[key] - before[key] for key in METRICS}
    return output, {
        "ttft_s": None if first is None else first - start,
        "wall_s": end - start, "usage": usage, "finish_reason": finish_reason,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-set", choices=("32k", "128k", "199k"), default="32k")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    all_tasks = {"32k": tasks_32k, "128k": tasks_128k,
                 "199k": tasks_199k}[args.task_set]()
    if args.limit is not None and not 1 <= args.limit <= len(all_tasks):
        parser.error(f"limit must be between 1 and {len(all_tasks)}")
    selected = all_tasks[:args.limit]
    previous = []
    if args.output.exists():
        previous = [json.loads(line) for line in args.output.read_text().splitlines()]
        if any(row["arm"] != args.arm for row in previous):
            raise ValueError("output file contains a different arm")
    completed = {row["id"]: row for row in previous}
    image = subprocess.check_output(
        ["docker", "image", "inspect", "local/b70-qwen38-vllm:onednn-poc-20260929",
         "--format", "{{.Id}}"], text=True).strip()
    for task in selected:
        prompt_hash = hashlib.sha256(task["prompt"].encode()).hexdigest()
        if task["id"] in completed:
            if completed[task["id"]]["prompt_sha256"] != prompt_hash:
                raise ValueError("resumed task prompt changed")
            continue
        output, timing = request(args.base, task["prompt"])
        judgement = score(task, output)
        row = {
            "id": task["id"], "kind": task["kind"], "arm": args.arm,
            "image": image, "context_sha256": task["context_sha256"],
            "prompt_sha256": prompt_hash,
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "output": output, "score": judgement, **timing,
        }
        with args.output.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps({key: value for key, value in row.items()
                          if key != "output"}), flush=True)


if __name__ == "__main__":
    main()

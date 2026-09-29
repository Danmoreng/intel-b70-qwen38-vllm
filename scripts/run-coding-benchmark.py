#!/usr/bin/env python3
"""Repeatable two-stage agentic coding benchmark against the live local engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request


REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "benchmarks/coding-fixture/v1"
MODEL = "Qwen3.8-27B"
METRICS = {
    "prompt_tokens": "vllm:prompt_tokens_total",
    "cached_tokens": "vllm:prompt_tokens_cached_total",
    "prefill_tokens": "vllm:request_prefill_kv_computed_tokens_sum",
    "prefill_seconds": "vllm:request_prefill_time_seconds_sum",
    "decode_seconds": "vllm:request_decode_time_seconds_sum",
    "generation_tokens": "vllm:generation_tokens_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "preemptions": "vllm:num_preemptions_total",
    "completed": "vllm:request_success_total",
}
TOOLS = [
    {"type": "function", "function": {"name": "list_files",
      "description": "List every file in the QueueKit project.",
      "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "read_file",
      "description": "Read a project file by relative path.",
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                     "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "write_file",
      "description": "Replace one project file with UTF-8 content. Use this for code edits.",
      "parameters": {"type": "object", "properties": {
          "path": {"type": "string"}, "content": {"type": "string"}},
          "required": ["path", "content"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "run_tests",
      "description": "Run the visible unittest suite in the project.",
      "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and
                       "__pycache__" not in p.parts):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_fixture():
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    expected = manifest["files"]
    actual = {str(p.relative_to(FIXTURE)): sha(p)
              for p in FIXTURE.rglob("*") if p.is_file() and
              p.name != "manifest.json" and "__pycache__" not in p.parts}
    if actual != expected:
        raise RuntimeError("coding fixture differs from its frozen manifest")
    return sha(FIXTURE / "manifest.json")


def http_json(url: str, payload=None, timeout=60):
    req = urllib.request.Request(
        url, data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def snapshot(base: str):
    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
        raw = response.read().decode()
    result = {key: 0.0 for key in METRICS}
    for line in raw.splitlines():
        match = re.match(r"([^\s{]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if match:
            for key, name in METRICS.items():
                if match.group(1) == name:
                    result[key] += float(match.group(2))
    return result


def live_identity(container: str):
    raw = subprocess.check_output(["docker", "inspect", container], text=True)
    item = json.loads(raw)[0]
    return {"container_id": item["Id"], "image_id": item["Image"],
            "image_tag": item["Config"]["Image"],
            "command": item["Config"]["Cmd"],
            "policy_sha256": item["Config"]["Labels"].get(
                "org.local.b70.policy.sha256")}


def project_path(project: Path, name: str) -> Path:
    root = project.resolve()
    target = (project / name).resolve()
    if (target == root or root not in target.parents or
            any(part in ("__pycache__", ".git") for part in target.parts)):
        raise ValueError("path outside the project")
    return target


def execute_tool(project: Path, name: str, arguments: dict):
    try:
        if name == "list_files":
            return {"files": [str(p.relative_to(project)) for p in sorted(project.rglob("*"))
                              if p.is_file() and "__pycache__" not in p.parts]}
        if name == "read_file":
            path = project_path(project, arguments["path"])
            return {"path": arguments["path"], "content": path.read_text()[:50000]}
        if name == "write_file":
            path = project_path(project, arguments["path"])
            content = arguments["content"]
            if not isinstance(content, str) or len(content) > 100000:
                raise ValueError("invalid file content")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return {"path": arguments["path"], "sha256": sha(path)}
        if name == "run_tests":
            done = subprocess.run([sys.executable, "-m", "unittest", "discover",
                                   "-s", "tests", "-v"], cwd=project,
                                  capture_output=True, text=True, timeout=30)
            return {"exit_code": done.returncode,
                    "output": (done.stdout + done.stderr)[-10000:]}
        raise ValueError(f"unknown tool {name}")
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def acceptance(project: Path, task_path: Path):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project)
    done = subprocess.run([sys.executable, "-m", "unittest", "-v", str(task_path)],
                          cwd=project, env=environment, capture_output=True,
                          text=True, timeout=30)
    return {"passed": done.returncode == 0, "exit_code": done.returncode,
            "output": (done.stdout + done.stderr)[-12000:]}


def run(args):
    manifest_sha = verify_fixture()
    tasks = json.loads((FIXTURE / "tasks.json").read_text())["tasks"]
    expected_policy = (REPO / "config/production_policy.sha256").read_text().split()[0]
    identity = live_identity(args.container)
    if identity["policy_sha256"] != expected_policy:
        raise RuntimeError("live image does not have the production policy")
    with urllib.request.urlopen(args.base + "/health", timeout=10) as response:
        if response.status != 200:
            raise RuntimeError("engine is not healthy")
    if args.output_root.exists():
        raise ValueError("output directory already exists; use a fresh run")
    args.output_root.mkdir(parents=True)
    project = args.output_root / "project"
    shutil.copytree(FIXTURE / "project", project)
    original_hash = tree_hash(FIXTURE / "project")
    if tree_hash(project) != original_hash:
        raise RuntimeError("fixture copy changed")
    start = time.monotonic()
    before = snapshot(args.base)
    messages = [{"role": "system", "content":
        "You are a software engineer working in the QueueKit project. Use the tools to inspect files, edit code, and run visible tests. Make the smallest complete implementation. When done with a task, send a concise final message without tool calls. Do not assume hidden acceptance tests are visible."}]
    records = []
    results = []
    for task in tasks:
        messages.append({"role": "user", "content": task["instruction"]})
        for step in range(args.max_requests_per_task):
            payload = {"model": MODEL, "messages": messages, "tools": TOOLS,
                       "tool_choice": "auto", "temperature": 1.0,
                       "top_p": 0.95, "top_k": 20, "seed": 7000 + len(records),
                       "max_tokens": 4096,
                       "chat_template_kwargs": {"enable_thinking": True}}
            started = time.monotonic()
            answer = http_json(args.base + "/v1/chat/completions", payload,
                               timeout=900)
            finished = time.monotonic()
            choice = answer["choices"][0]
            message = choice["message"]
            calls = message.get("tool_calls") or []
            record = {"task": task["id"], "step": step + 1,
                      "wall_s": finished - started,
                      "usage": answer.get("usage"),
                      "finish_reason": choice.get("finish_reason"),
                      "tool_names": [call["function"]["name"] for call in calls],
                      "response_sha256": hashlib.sha256(
                          json.dumps(message, sort_keys=True).encode()).hexdigest()}
            records.append(record)
            messages.append(message)
            for call in calls:
                try:
                    parameters = json.loads(call["function"].get("arguments") or "{}")
                    output = execute_tool(project, call["function"]["name"], parameters)
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    output = {"error": f"{type(error).__name__}: {error}"}
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(output)})
            with (args.output_root / "requests.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps({"task": task["id"], "step": step + 1,
                              "wall_s": round(record["wall_s"], 2),
                              "tools": record["tool_names"]}), flush=True)
            if not calls and choice.get("finish_reason") != "length":
                break
            if not calls and choice.get("finish_reason") == "length":
                messages.append({"role": "user", "content":
                                 "Continue the implementation and use tools as needed."})
        check = acceptance(project, FIXTURE / task["acceptance"])
        results.append({"task": task["id"], "acceptance": check,
                        "project_sha256": tree_hash(project)})
        print(json.dumps({"task": task["id"], "passed": check["passed"]}),
              flush=True)
    after = snapshot(args.base)
    deltas = {key: after[key] - before[key] for key in METRICS}
    summary = {"fixture_id": "queuekit-agent-v1",
               "fixture_manifest_sha256": manifest_sha,
               "fixture_project_sha256": original_hash,
               "tasks_sha256": sha(FIXTURE / "tasks.json"),
               "identity": identity,
               "wall_s": time.monotonic() - start,
               "requests": len(records),
               "tool_calls": sum(len(row["tool_names"]) for row in records),
               "task_results": results,
               "metrics": deltas,
               "prompt_tokens_by_usage": sum((row["usage"] or {}).get("prompt_tokens", 0)
                                             for row in records),
               "generated_tokens_by_usage": sum((row["usage"] or {}).get("completion_tokens", 0)
                                                for row in records)}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "messages.json").write_text(json.dumps(messages, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "task_results"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--container", default="b70-qwen38-vllm")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-requests-per-task", type=int, default=40)
    run(parser.parse_args())

#!/usr/bin/env python3
"""Frozen, engine-neutral coding loop with per-request native phase accounting."""

from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import platform
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "benchmarks/web-coding-harness"
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
SYSTEM = """You are implementing the supplied small web application in a local project. Inspect the README and relevant public specs/source, implement the requested stage, and use run_tests plus browser_test to validate it. The supplied visible contract checks and real software browser are your primary test tools. You are not required to create a separate test suite. Do not build DOM or WebGL mocks; use the actual browser tool for those behaviors. If a pure-module behavior needs an extra regression case, add at most three small cases per module. Do not invent stricter requirements or exception types than the public spec. Keep earlier contracts working, preserve protected specs/package/harness, and use exact-span edit_file for small corrections. Prefer relevant read_file line ranges rather than whole-source audits. No shell, internet or third-party runtime libraries. Do not add features or pad the conversation. After a stage passes its relevant supplied checks, send a concise final message without tool calls. More stages may follow in the same conversation."""


def tool(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    tool("list_files", "List all project files."),
    tool(
        "read_file",
        "Read a UTF-8 project file, optionally by 1-indexed inclusive line range.",
        {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        },
        ["path"],
    ),
    tool(
        "write_file",
        "Replace an editable project file with UTF-8 content.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    tool(
        "edit_file",
        "Replace one exact, unique nonempty text span in an editable file. The old_text must match exactly once.",
        {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
        },
        ["path", "old_text", "new_text"],
    ),
    tool(
        "run_tests",
        "Run visible contract checks through the current stage and your own Node regression tests.",
    ),
    tool(
        "browser_test",
        "Test WebGL2 rendering and app functionality in a pinned, software-rendered browser. Browser is closed before the next model request.",
    ),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def tree(root):
    return {
        str(p.relative_to(root)): sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and "node_modules" not in p.parts
    }


def http(base, path, payload=None, timeout=30):
    req = urllib.request.Request(
        base + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def bound_output(base, payload):
    """Render the actual retained history before reserving output capacity."""
    started = time.monotonic()
    messages = []
    for original in payload["messages"]:
        message = dict(original)
        # ChatCompletionRequest normalizes this alias; TokenizeChatRequest does
        # not. Without this copy, /tokenize silently omits prior reasoning.
        reasoning = message.pop("reasoning_content", None)
        if reasoning is not None and message.get("reasoning") is None:
            message["reasoning"] = reasoning
        messages.append(message)
    rendered = http(
        base,
        "/tokenize",
        {
            "model": payload["model"],
            "messages": messages,
            "tools": payload["tools"],
            "chat_template_kwargs": payload["chat_template_kwargs"],
            "add_generation_prompt": True,
        },
        timeout=60,
    )
    count = rendered["count"]
    context_limit = min(200704, rendered["max_model_len"])
    if count >= context_limit:
        raise RuntimeError("common context window exhausted: " + str(count))
    payload["max_tokens"] = min(payload["max_tokens"], context_limit - count)
    return {
        "prompt_tokens": count,
        "context_limit": context_limit,
        "max_output_tokens": payload["max_tokens"],
        "wall_s": time.monotonic() - started,
    }


def snapshot(base):
    with urllib.request.urlopen(base + "/metrics", timeout=15) as r:
        raw = r.read().decode()
    result = {}
    for line in raw.splitlines():
        m = re.match(r"([^\s{]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if m:
            for k, name in METRICS.items():
                if m[1] == name:
                    result[k] = result.get(k, 0) + float(m[2])
    return result


def delta(a, b):
    return {k: a[k] - b[k] for k in a.keys() & b.keys()}


def wait_accounted(base, before, usage):
    deadline = time.monotonic() + 45
    while True:
        after = snapshot(base)
        d = delta(after, before)
        if (
            d.get("completed") == 1
            and d.get("generation_tokens") == usage.get("completion_tokens")
            and d.get("prompt_tokens") == usage.get("prompt_tokens")
        ):
            return after, d
        if d.get("completed", 0) > 1 or d.get("generation_tokens", 0) > usage.get(
            "completion_tokens", 0
        ):
            raise RuntimeError(
                "concurrent traffic or inconsistent request accounting: " + str(d)
            )
        if time.monotonic() > deadline:
            raise RuntimeError("native accounting did not settle: " + str(d))
        time.sleep(0.25)


def energy():
    p = list(
        Path("/sys/bus/pci/devices/0000:03:00.0/hwmon").glob("hwmon*/energy1_input")
    )
    return int(p[0].read_text()) if len(p) == 1 else None


def cap():
    p = list(Path("/sys/bus/pci/devices/0000:03:00.0/hwmon").glob("hwmon*/power1_cap"))
    if len(p) != 1 or int(p[0].read_text()) != 180000000:
        raise RuntimeError("B70 power cap must be 180W")


def identity(container):
    x = json.loads(
        subprocess.check_output(["docker", "inspect", container], text=True)
    )[0]
    return {
        k: v
        for k, v in {
            "container_id": x["Id"],
            "image_id": x["Image"],
            "image_tag": x["Config"]["Image"],
            "command": x["Config"]["Cmd"],
            "labels": x["Config"]["Labels"],
        }.items()
    }


def stream(base, payload, sse_path):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first = None
    usage = None
    finish = None
    content = []
    reasoning = []
    calls = {}
    with (
        urllib.request.urlopen(req, timeout=1800) as response,
        sse_path.open("w") as out,
    ):
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            if raw.strip() == b"data: [DONE]":
                break
            event = json.loads(raw[6:])
            out.write(
                json.dumps(
                    {"elapsed_s": time.monotonic() - started, "event": event},
                    ensure_ascii=False,
                )
                + "\n"
            )
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                d = choice.get("delta") or {}
                finish = choice.get("finish_reason") or finish
                signal = bool(
                    d.get("content")
                    or d.get("reasoning_content")
                    or d.get("reasoning")
                    or d.get("tool_calls")
                )
                if signal and first is None:
                    first = time.monotonic() - started
                if d.get("content"):
                    content.append(d["content"])
                if d.get("reasoning_content") or d.get("reasoning"):
                    reasoning.append(d.get("reasoning_content") or d["reasoning"])
                for t in d.get("tool_calls") or []:
                    c = calls.setdefault(
                        t["index"],
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if t.get("id"):
                        c["id"] += t["id"]
                    f = t.get("function") or {}
                    if f.get("name"):
                        c["function"]["name"] += f["name"]
                    if f.get("arguments"):
                        c["function"]["arguments"] += f["arguments"]
    message = {"role": "assistant", "content": "".join(content) or None}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    # Both Qwen templates support retained reasoning_content; make this explicit
    # instead of silently dropping the model's working history between tool turns.
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    return {
        "message": message,
        "reasoning": "".join(reasoning),
        "usage": usage,
        "finish_reason": finish,
        "wall_s": time.monotonic() - started,
        "ttft_s": first,
    }


def isolated_node(project, arguments, readonly=False):
    """Execute generated Node code with only the project and grader mounted."""
    if not shutil.which("bwrap"):
        raise RuntimeError("bubblewrap is required for isolated project tests")
    grader_mount = ["--ro-bind", str(HARNESS.resolve()), "/harness"] if readonly else []
    return [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib",
        "/lib64",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        *grader_mount,
        "--ro-bind" if readonly else "--bind",
        str(project.resolve()),
        "/project",
        "--chdir",
        "/project",
        "--clearenv",
        "--setenv",
        "PATH",
        "/usr/bin",
        "--setenv",
        "LANG",
        "C.UTF-8",
        "--setenv",
        "TZ",
        "UTC",
        "--setenv",
        "HOME",
        "/tmp",
        "--",
        "/usr/bin/node",
        *arguments,
    ]


def run_node(script, project, stage, *extra):
    argv = (
        isolated_node(
            project,
            ["/harness/" + script, "/project", str(stage), *map(str, extra)],
            readonly=True,
        )
        if script == "checks.mjs"
        else ["node", str(HARNESS / script), str(project), str(stage), *map(str, extra)]
    )
    p = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=75,
    )
    try:
        x = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        x = {"harness_error": (p.stdout + p.stderr)[-8000:]}
    return {"exit_code": p.returncode, **x}


def safe_path(project, name):
    target = (project / name).resolve()
    if project.resolve() not in target.parents or target.is_symlink():
        raise ValueError("path outside project")
    return target


def execute(project, name, params, stage, output, index):
    try:
        if name == "list_files":
            return {"files": sorted(tree(project))}
        if name == "read_file":
            p = safe_path(project, params["path"])
            lines = p.read_text().splitlines(keepends=True)
            start = params.get("start_line", 1)
            end = params.get("end_line", len(lines))
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start < 1
                or end < start
            ):
                raise ValueError("invalid line range")
            text = "".join(lines[start - 1 : end])
            return {
                "path": params["path"],
                "start_line": start,
                "end_line": min(end, len(lines)),
                "total_lines": len(lines),
                "content": text[:60000],
                "truncated": len(text) > 60000,
            }
        if name in ("write_file", "edit_file"):
            p = safe_path(project, params["path"])
            rel = p.relative_to(project).as_posix()
            if not (
                rel.startswith("src/")
                and rel.endswith(".js")
                or rel.startswith("tests/")
                and rel.endswith(".test.js")
                or rel in ("index.html", "main.js", "style.css", "USER_GUIDE.md")
            ):
                raise ValueError("protected path")
            if name == "write_file":
                v = params["content"]
            else:
                old, new = params["old_text"], params["new_text"]
                original = p.read_text()
                if (
                    not isinstance(old, str)
                    or not old
                    or not isinstance(new, str)
                    or original.count(old) != 1
                ):
                    raise ValueError(
                        "old_text must match exactly once; no changes made"
                    )
                v = original.replace(old, new, 1)
            if not isinstance(v, str) or len(v) > 150000:
                raise ValueError("invalid content")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(v)
            return {"path": rel, "sha256": sha(p), "bytes": len(v.encode())}
        if name == "run_tests":
            contracts = run_node("checks.mjs", project, stage)
            tests = sorted(str(p) for p in (project / "tests").glob("*.test.js"))
            if not tests:
                return {
                    "contracts": contracts,
                    "regression_tests": {
                        "exit_code": 0,
                        "output": "No agent-written regression tests yet. Independent visible checks ran above.",
                    },
                }
            done = subprocess.run(
                isolated_node(
                    project,
                    [
                        "--test",
                        *[
                            "/project/" + str(Path(p).relative_to(project))
                            for p in tests
                        ],
                    ],
                ),
                cwd=project,
                capture_output=True,
                text=True,
                timeout=45,
            )
            return {
                "contracts": contracts,
                "regression_tests": {
                    "exit_code": done.returncode,
                    "output": (done.stdout + done.stderr)[-9000:],
                },
            }
        if name == "browser_test":
            return run_node(
                "browser.mjs", project, stage, output / f"browser-{index:04d}.png"
            )
        raise ValueError("unknown tool " + name)
    except Exception as e:
        return {"error": type(e).__name__ + ": " + str(e)}


def normalize_tool_result(value, project):
    """Remove observational timing/path noise from the model's tool history."""
    if isinstance(value, dict):
        return {k: normalize_tool_result(v, project) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_tool_result(v, project) for v in value]
    if not isinstance(value, str):
        return value
    value = (
        value.replace(str(project.resolve()), "/project")
        .replace(str(HARNESS.resolve()), "/harness")
        .replace(str(REPO.resolve()), "/benchmark")
    )
    value = re.sub(r"\(\d+(?:\.\d+)?ms\)", "(<time>ms)", value)
    value = re.sub(r"(duration_ms\s+)\d+(?:\.\d+)?", r"\1<time>", value)
    return value


def summarize(records):
    def group(rs):
        native = {k: sum(r["native"].get(k, 0) for r in rs) for k in METRICS}
        post_first = sum(max(0, r["usage"]["completion_tokens"] - 1) for r in rs)
        client_rows = [r for r in rs if r["ttft_s"] is not None]
        client_window = sum(max(0, r["wall_s"] - r["ttft_s"]) for r in client_rows)
        client_post_first = sum(
            max(0, r["usage"]["completion_tokens"] - 1) for r in client_rows
        )
        return {
            "requests": len(rs),
            "context_min": min((r["usage"]["prompt_tokens"] for r in rs), default=None),
            "context_max": max((r["usage"]["prompt_tokens"] for r in rs), default=None),
            "native": native,
            "post_first_tokens": post_first,
            "prefill_tps": native["prefill_tokens"] / native["prefill_seconds"]
            if native["prefill_seconds"]
            else None,
            "decode_tps": post_first / native["decode_seconds"]
            if native["decode_seconds"]
            else None,
            "cached_fraction": native["cached_tokens"] / native["prompt_tokens"]
            if native["prompt_tokens"]
            else None,
            "mtp_acceptance": native["accepted_tokens"] / native["draft_tokens"]
            if native["draft_tokens"]
            else None,
            "client_request_seconds": sum(r["wall_s"] for r in rs),
            "client_decode_tps": client_post_first / client_window
            if client_window > 0
            else None,
        }

    bands = {}
    for r in records:
        bands.setdefault(r["usage"]["prompt_tokens"] // 10000, []).append(r)
    return {
        "overall": group(records),
        "bands": [
            {"lower": i * 10000, "upper": (i + 1) * 10000, **group(bands.get(i, []))}
            for i in range(max(bands, default=0) + 1)
        ],
    }


def run(args):
    global HARNESS
    fixture = args.fixture.resolve()
    manifest = json.loads((fixture / "manifest.json").read_text())
    if manifest.get("runner_sha256") and manifest["runner_sha256"] != sha(
        Path(__file__)
    ):
        raise RuntimeError("runner differs from the frozen fixture version")
    if manifest.get("harness_version"):
        HARNESS = HARNESS / manifest["harness_version"]
    actual = tree(fixture)
    actual.pop("manifest.json", None)
    if actual != manifest["files"]:
        raise RuntimeError("fixture differs from frozen manifest")
    harness_files = {
        f: sha(HARNESS / f)
        for f in ["checks.mjs", "browser.mjs", "package.json", "package-lock.json"]
    }
    if harness_files != manifest["harness_files"]:
        raise RuntimeError("harness differs from frozen manifest")
    if args.output.exists():
        raise RuntimeError("fresh output directory required")
    models = http(args.base, "/v1/models")
    model = args.model or models["data"][0]["id"]
    ident = identity(args.container)
    cap()
    required = {
        "completed",
        "prompt_tokens",
        "cached_tokens",
        "prefill_tokens",
        "prefill_seconds",
        "decode_seconds",
        "generation_tokens",
        "preemptions",
    }
    if not required <= snapshot(args.base).keys():
        raise RuntimeError("missing native metrics")
    args.output.mkdir(parents=True)
    shutil.copyfile(Path(__file__), args.output / "runner-source.py")
    project = args.output / "project"
    shutil.copytree(fixture / "project", project)
    tasks = json.loads((fixture / "tasks.json").read_text())["tasks"][: args.stages]
    messages = [{"role": "system", "content": SYSTEM}]
    records = []
    stages = []
    started = time.monotonic()
    energy_start = energy()
    tool_s = 0
    provenance = {
        "fixture_id": manifest["fixture_id"],
        "fixture_manifest_sha256": sha(fixture / "manifest.json"),
        "fixture_files": manifest["files"],
        "harness_files": harness_files,
        "runner_sha256": sha(Path(__file__)),
        "identity": ident,
        "model_info": models,
        "node_version": subprocess.check_output(
            ["node", "--version"], text=True
        ).strip(),
        "started_at": now(),
        "sampling": {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 20,
            "seed_base": args.seed,
            "max_tokens": 16384,
            "enable_thinking": True,
            "reasoning_effort": "medium",
            "preserve_thinking": True,
            "thinking_token_budget": args.thinking_token_budget,
            "output_budget_policy": "at most 16384; rendered /tokenize history with reasoning alias normalized; input+output fits common 200704-token window or smaller engine limit",
        },
        "limits": {
            "max_requests_per_stage": args.max_requests_per_stage,
            "max_wall_seconds": args.max_wall_seconds,
            "stages": len(tasks),
        },
        "measurement": "exclusive C1, native per-request accounting; prefix cache stays enabled; first generated token excluded from decode; 10K=10000 input-token bands",
    }
    provenance["host"] = {
        "kernel": platform.release(),
        "machine": platform.machine(),
        "power_cap_w": 180,
        "runtime_source_files": tree(REPO / "runtime"),
        "production_policy_sha256": sha(REPO / "config/production_policy.json"),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
    }
    provenance["tool_output_normalization"] = (
        "Incidental Node test timings and absolute host/project paths normalized in agent history; unmodified raw tool results archived."
    )
    provenance["test_isolation"] = (
        "Node checks/tests: bubblewrap, project-only filesystem, no host home/network/GPU; independent grading project read-only. Browser: SwiftShader software WebGL2, external requests blocked."
    )
    save(args.output / "provenance.json", provenance)
    status = "running"
    error = None
    try:
        for number, task in enumerate(tasks, 1):
            messages.append(
                {
                    "role": "user",
                    "content": task["instruction"]
                    + "\nApplicable contracts: "
                    + ", ".join("specs/" + m + ".md" for m in task["modules"]),
                }
            )
            completed = False
            for step in range(args.max_requests_per_stage):
                if time.monotonic() - started > args.max_wall_seconds:
                    raise RuntimeError("run wall-time limit")
                cap()
                index = len(records) + 1
                before = snapshot(args.base)
                request_energy = energy()
                payload = {
                    "model": model,
                    "messages": messages,
                    "tools": TOOLS,
                    "tool_choice": "auto",
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "seed": args.seed + index - 1,
                    "max_tokens": 16384,
                    "thinking_token_budget": args.thinking_token_budget,
                    "reasoning_effort": "medium",
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                        "reasoning_effort": "medium",
                        "preserve_thinking": True,
                    },
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                preflight = bound_output(args.base, payload)
                save(args.output / f"request-{index:04d}.json", payload)
                save(
                    args.output / "state.json",
                    {
                        "status": "running",
                        "stage": task["id"],
                        "request": index,
                        "last_context": records[-1]["usage"]["prompt_tokens"]
                        if records
                        else None,
                        "updated_at": now(),
                    },
                )
                answer = stream(
                    args.base, payload, args.output / f"response-{index:04d}.sse.jsonl"
                )
                save(args.output / f"response-{index:04d}.json", answer)
                if not answer["usage"]:
                    raise RuntimeError("stream omitted usage")
                if answer["usage"]["prompt_tokens"] != preflight["prompt_tokens"]:
                    raise RuntimeError(
                        "tokenize preflight differs from inference input"
                    )
                after, native = wait_accounted(args.base, before, answer["usage"])
                cap()
                if identity(args.container) != ident:
                    raise RuntimeError("worker identity changed")
                message = answer["message"]
                calls = message.get("tool_calls") or []
                rec = {
                    "request": index,
                    "stage": task["id"],
                    "step": step + 1,
                    "usage": answer["usage"],
                    "finish_reason": answer["finish_reason"],
                    "wall_s": answer["wall_s"],
                    "ttft_s": answer["ttft_s"],
                    "native": native,
                    "native_before": before,
                    "native_after": after,
                    "tokenize_preflight": preflight,
                    "card_energy_j": (energy() - request_energy) / 1e6
                    if request_energy is not None
                    else None,
                    "tools": [c["function"]["name"] for c in calls],
                    "request_sha256": sha(args.output / f"request-{index:04d}.json"),
                    "response_sha256": sha(args.output / f"response-{index:04d}.json"),
                }
                records.append(rec)
                with (args.output / "requests.jsonl").open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                messages.append(message)
                print(
                    json.dumps(
                        {
                            "request": index,
                            "stage": task["id"],
                            "context": answer["usage"]["prompt_tokens"],
                            "generated": answer["usage"]["completion_tokens"],
                            "wall_s": round(answer["wall_s"], 2),
                            "tools": rec["tools"],
                            "decode_tps": round(
                                max(0, answer["usage"]["completion_tokens"] - 1)
                                / native["decode_seconds"],
                                2,
                            )
                            if native["decode_seconds"]
                            else None,
                        }
                    ),
                    flush=True,
                )
                for call in calls:
                    t = time.monotonic()
                    try:
                        params = json.loads(call["function"].get("arguments") or "{}")
                        result = execute(
                            project,
                            call["function"]["name"],
                            params,
                            number,
                            args.output,
                            index,
                        )
                    except Exception as e:
                        result = {"error": type(e).__name__ + ": " + str(e)}
                    tool_s += time.monotonic() - t
                    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", call["id"])[:60]
                    agent_result = normalize_tool_result(result, project)
                    save(
                        args.output / f"tool-{index:04d}-{safe_id}.json",
                        {
                            "raw_result": result,
                            "agent_result": agent_result,
                            "tool_wall_s": time.monotonic() - t,
                        },
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(agent_result, ensure_ascii=False),
                        }
                    )
                save(args.output / "conversation.json", messages)
                if not calls and answer["finish_reason"] != "length":
                    completed = True
                    break
                if not calls:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Continue the current stage using tools and complete the implementation.",
                        }
                    )
            check = run_node("checks.mjs", project, number, "--acceptance")
            browser = (
                run_node(
                    "browser.mjs",
                    project,
                    number,
                    args.output / f"stage-{number:02d}.png",
                )
                if number >= 4
                else None
            )
            stages.append(
                {
                    "stage": task["id"],
                    "agent_declared_complete": completed,
                    "acceptance": check,
                    "browser": browser,
                    "project_sha256": hashlib.sha256(
                        json.dumps(tree(project), sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
            save(args.output / "stages.json", stages)
            shutil.copytree(project, args.output / f"project-after-{task['id']}")
            print(
                json.dumps(
                    {
                        "stage_complete": task["id"],
                        "acceptance_passed": check.get("passed"),
                        "acceptance_total": check.get("total"),
                        "browser_passed": browser.get("passed") if browser else None,
                    }
                ),
                flush=True,
            )
        status = "complete"
    except BaseException as e:
        status = "failed"
        error = type(e).__name__ + ": " + str(e)
        raise
    finally:
        save(
            args.output / "summary.json",
            {
                "status": status,
                "error": error,
                "finished_at": now(),
                "fixture_id": manifest["fixture_id"],
                "fixture_manifest_sha256": sha(fixture / "manifest.json"),
                "runner_sha256": provenance["runner_sha256"],
                "identity": ident,
                "wall_s": time.monotonic() - started,
                "tool_s": tool_s,
                "card_energy_wh": (energy() - energy_start) / 3.6e9
                if energy_start is not None
                else None,
                "tool_calls": sum(len(r["tools"]) for r in records),
                "requests": len(records),
                "stages": stages,
                **summarize(records),
            },
        )
        save(
            args.output / "state.json",
            {"status": status, "error": error, "updated_at": now()},
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base", default="http://127.0.0.1:8081")
    p.add_argument("--container", default="b70-qwen38-vllm")
    p.add_argument("--model")
    p.add_argument("--seed", type=int, default=73000)
    p.add_argument("--stages", type=int, default=100)
    p.add_argument("--max-requests-per-stage", type=int, default=32)
    p.add_argument("--max-wall-seconds", type=int, default=7200)
    p.add_argument("--thinking-token-budget", type=int, default=4096)
    run(p.parse_args())


if __name__ == "__main__":
    main()

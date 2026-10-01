#!/usr/bin/env python3
"""Validate scored request accounting and export a public coding comparison."""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "web_runner", REPO / "scripts/run-web-coding-benchmark.py"
)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def load(p):
    return json.loads(p.read_text())


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)


def validate(root):
    summary = load(root / "summary.json")
    provenance = load(root / "provenance.json")
    reason = str(summary.get("error", ""))
    exhausted = summary["status"] == "failed" and (
        reason.startswith("RuntimeError: common context window exhausted:")
        or reason == "RuntimeError: run wall-time limit"
    )
    if summary["status"] != "complete" and not exhausted:
        raise RuntimeError("run has an unsupported measurement failure: " + str(root))
    if provenance["limits"]["stages"] != 6:
        raise RuntimeError("comparison requires the same six-stage assignment")
    if summary["status"] == "complete" and len(summary["stages"]) != 6:
        raise RuntimeError("complete run has missing task stages")
    rows = [json.loads(x) for x in (root / "requests.jsonl").read_text().splitlines()]
    if not rows or len(rows) != summary["requests"]:
        raise RuntimeError("request count differs")
    conversation = load(root / "conversation.json")
    assistant_positions = [
        i for i, m in enumerate(conversation) if m["role"] == "assistant"
    ]
    if len(assistant_positions) != len(rows):
        raise RuntimeError("conversation request count differs")
    for index, row in enumerate(rows, 1):
        if row["request"] != index:
            raise RuntimeError("request sequence differs")
        u = row["usage"]
        d = row["native"]
        for k, expected in [
            ("completed", 1),
            ("prompt_tokens", u["prompt_tokens"]),
            ("generation_tokens", u["completion_tokens"]),
        ]:
            if not close(d[k], expected):
                raise RuntimeError("native usage mismatch: " + k)
        if d["prefill_tokens"] + d["cached_tokens"] + 1e-6 < u["prompt_tokens"]:
            raise RuntimeError("missing prefill accounting")
        if d["prefill_seconds"] <= 0 or d["decode_seconds"] < 0:
            raise RuntimeError("invalid phase timing")
        if (
            R.sha(root / f"request-{index:04d}.json") != row["request_sha256"]
            or R.sha(root / f"response-{index:04d}.json") != row["response_sha256"]
        ):
            raise RuntimeError("raw request/response hash differs")
        request = load(root / f"request-{index:04d}.json")
        if request["messages"] != conversation[: assistant_positions[index - 1]]:
            raise RuntimeError("saved conversation differs from measured request")
        preflight = row["tokenize_preflight"]
        if (
            preflight["prompt_tokens"] != u["prompt_tokens"]
            or preflight["max_output_tokens"] != request["max_tokens"]
            or request["max_tokens"] > preflight["context_limit"] - u["prompt_tokens"]
            or u["completion_tokens"] > request["max_tokens"]
        ):
            raise RuntimeError("invalid input/output context reservation")
        for k, v in R.delta(row["native_after"], row["native_before"]).items():
            if not close(v, d[k]):
                raise RuntimeError("counter difference mismatch")
    recomputed = R.summarize(rows)
    if (
        recomputed["overall"] != summary["overall"]
        or recomputed["bands"] != summary["bands"]
    ):
        raise RuntimeError("weighted rate or band table differs")
    if (
        provenance["fixture_manifest_sha256"] != summary["fixture_manifest_sha256"]
        or provenance["identity"] != summary["identity"]
        or provenance["runner_sha256"] != summary["runner_sha256"]
    ):
        raise RuntimeError("provenance mismatch")
    # A context-budget failure is a task outcome, not a missing measurement.
    # Grade both stopped and completed outputs with the same frozen cases.
    final = load(root / "post-run-acceptance.json")
    if (
        final["fixture_manifest_sha256"] != provenance["fixture_manifest_sha256"]
        or final["harness_files"] != provenance["harness_files"]
    ):
        raise RuntimeError("post-run grading provenance differs")
    if final["acceptance"].get("harness_error") or (final.get("browser") or {}).get(
        "harness_error"
    ):
        raise RuntimeError("acceptance infrastructure failed")
    # The renderer identity row is metadata emitted only after successful render;
    # exclude it from functional scoring so every engine has the same denominator.
    cases = [{"id": "node/" + x["name"], **x} for x in final["acceptance"]["checks"]]
    cases += [
        {"id": "browser/" + x["name"], **x}
        for x in (final.get("browser") or {}).get("checks", [])
        if x["name"] != "browser renderer identity"
    ]
    quality = {
        "passed": sum(x["passed"] for x in cases),
        "total": len(cases),
        "checks": cases,
        "all_stages_declared_complete": len(summary["stages"]) == 6
        and all(x["agent_declared_complete"] for x in summary["stages"]),
    }
    if not cases:
        raise RuntimeError("no quality cases")
    manifests = [
        p
        for p in (REPO / "benchmarks/web-coding-fixture").glob("*/manifest.json")
        if R.sha(p) == provenance["fixture_manifest_sha256"]
    ]
    if len(manifests) != 1:
        raise RuntimeError("cannot identify the measured fixture")
    expected_cases = load(manifests[0]).get("expected_final_cases", 45)
    if (
        len(cases) != expected_cases
        or final.get("expected_final_cases", 45) != expected_cases
    ):
        raise RuntimeError("missing final acceptance cases")
    if summary["status"] == "complete" and R.tree(root / "project") != R.tree(
        root / "project-after-06-release"
    ):
        raise RuntimeError("final generated project changed after grading")
    project_hash = hashlib.sha256(
        json.dumps(R.tree(root / "project"), sort_keys=True).encode()
    ).hexdigest()
    if project_hash != final["project_sha256"]:
        raise RuntimeError("graded project hash differs")
    return summary, provenance, rows, quality


def number(v, d=1):
    return "—" if v is None else f"{v:,.{d}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gptq", type=Path, required=True)
    p.add_argument("--exl3", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    engines = {}
    records = {}
    for name, root in [("gptq", a.gptq), ("exl3", a.exl3)]:
        if root is None:
            continue
        s, prov, rows, q = validate(root)
        engines[name] = {
            "fixture_id": s["fixture_id"],
            "fixture_manifest_sha256": s["fixture_manifest_sha256"],
            "runner_sha256": s["runner_sha256"],
            "identity": s["identity"],
            "provenance": prov,
            "wall_s": s["wall_s"],
            "tool_s": s["tool_s"],
            "card_energy_wh": s["card_energy_wh"],
            "requests": s["requests"],
            "tool_calls": s["tool_calls"],
            "session_status": s["status"],
            "stop_reason": s.get("error"),
            "stages_graded_during_session": len(s["stages"]),
            "overall": s["overall"],
            "bands": s["bands"],
            "quality": q,
            "stage_completions": {
                "declared": sum(x["agent_declared_complete"] for x in s["stages"]),
                "total": prov["limits"]["stages"],
            },
            "project_files": R.tree(root / "project"),
            "conversation_sha256": R.sha(root / "conversation.json"),
            "raw_root": str(root.resolve()),
        }
        review_path = root / "post-run-contract-review.json"
        if review_path.exists():
            review = load(review_path)
            if review["project_sha256"] != load(root / "post-run-acceptance.json")[
                "project_sha256"
            ] or review["original_score"] != {
                "passed": q["passed"],
                "total": q["total"],
            }:
                raise RuntimeError("contract review differs from timed output")
            engines[name]["contract_review"] = review
        records[name] = rows
    if len(engines) == 2:
        g, e = engines["gptq"], engines["exl3"]
        for k in ["fixture_manifest_sha256", "runner_sha256"]:
            if g[k] != e[k]:
                raise RuntimeError("engine comparison differs in " + k)
        for k in ["sampling", "harness_files", "limits"]:
            if g["provenance"][k] != e["provenance"][k]:
                raise RuntimeError("engine comparison differs in " + k)
        if [x["id"] for x in g["quality"]["checks"]] != [
            x["id"] for x in e["quality"]["checks"]
        ]:
            raise RuntimeError("different quality denominators")
    result = {
        "schema_version": 1,
        "created_at": R.now(),
        "comparison_scope": "Two complete serving profiles and quantized checkpoints; adaptive coding histories; one fixed seed. Not an isolated quantization or engine-kernel experiment.",
        "measurement": {
            "context_bands": "10000 tokens, lower inclusive, upper exclusive, full rendered input including retained reasoning and tool history",
            "prefill": "sum(native newly computed KV tokens)/sum(native prefill seconds); cached tokens separate",
            "decode": "sum(max(completion_tokens-1,0))/sum(native decode seconds), includes generated reasoning",
            "cache": "fresh worker before excluded warmup; prefix caching enabled throughout each session",
            "quality": "independent final API/physics/input/WebGL/browser checks; renderer identity is metadata, not a quality case",
        },
        "engines": engines,
    }
    a.output.mkdir(parents=True, exist_ok=True)
    R.save(a.output / "comparison.json", result)
    for name, root in [("gptq", a.gptq), ("exl3", a.exl3)]:
        if root is None:
            continue
        R.save(a.output / (name + "-requests.json"), records[name])
        shutil.copyfile(
            root / "post-run-acceptance.json", a.output / (name + "-acceptance.json")
        )
        if (root / "post-run-contract-review.json").exists():
            shutil.copyfile(
                root / "post-run-contract-review.json",
                a.output / (name + "-contract-review.json"),
            )
        # The task is public and tool history already normalizes host paths.
        # One compressed final transcript captures all adaptive request prefixes
        # without publishing the much larger repeated payload/SSE archive.
        transcript = (root / "conversation.json").read_bytes()
        compressed = gzip.compress(transcript, mtime=0)
        (a.output / (name + "-conversation.json.gz")).write_bytes(compressed)
        target = a.output / (name + "-project")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(root / "project", target)
        if (root / "post-run.png").exists():
            shutil.copyfile(root / "post-run.png", a.output / (name + "-game.png"))
    lines = [
        "# Reproducible Flappy Bird coding benchmark",
        "",
        "One fixed WebGL2 game task, one continuous conversation per engine, independent functional acceptance checks. No replay system or level editor.",
        "",
        "| Metric | GPTQ production v2 | EXL3 4.00 bpw |",
        "|---|---:|---:|",
    ]
    for label, key in [
        ("Wall time (minutes)", "wall_s"),
        ("Requests", "requests"),
        ("Tool calls", "tool_calls"),
        ("Tool execution (minutes)", "tool_s"),
        ("Card energy (Wh)", "card_energy_wh"),
    ]:
        vals = []
        for name in ["gptq", "exl3"]:
            v = engines.get(name, {}).get(key)
            vals.append(
                number(
                    v / 60 if v is not None and key in ["wall_s", "tool_s"] else v,
                    0 if key in ["requests", "tool_calls"] else 1,
                )
            )
        lines.append("| " + label + " | " + " | ".join(vals) + " |")
    for label, key in [
        ("Maximum context", "context_max"),
        ("Native prefill (tok/s)", "prefill_tps"),
        ("Native decode (tok/s)", "decode_tps"),
    ]:
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                number(
                    engines.get(n, {}).get("overall", {}).get(key),
                    0 if key == "context_max" else 1,
                )
                for n in ["gptq", "exl3"]
            )
            + " |"
        )
    lines.append(
        "| Session outcome | "
        + " | ".join(
            "Task budget exhausted; task incomplete"
            if engines[n]["session_status"] == "failed"
            else "All six stages attempted"
            for n in ["gptq", "exl3"]
            if n in engines
        )
        + " |"
    )
    lines.append(
        "| Functional checks | "
        + " | ".join(
            f"{engines[n]['quality']['passed']}/{engines[n]['quality']['total']}"
            if n in engines
            else "—"
            for n in ["gptq", "exl3"]
        )
        + " |"
    )
    if all("contract_review" in x for x in engines.values()):
        lines.append(
            "| Checks after public-contract review | "
            + " | ".join(
                f"{engines[n]['contract_review']['passed']}/{engines[n]['contract_review']['total']}"
                for n in ["gptq", "exl3"]
                if n in engines
            )
            + " |"
        )
    lines.append(
        "| Stages agent declared complete | "
        + " | ".join(
            f"{engines[n]['stage_completions']['declared']}/{engines[n]['stage_completions']['total']}"
            if n in engines
            else "—"
            for n in ["gptq", "exl3"]
        )
        + " |"
    )
    for label, key in [
        ("Logical input tokens", "prompt_tokens"),
        ("Generated tokens (including reasoning)", "generation_tokens"),
        ("Newly computed KV tokens", "prefill_tokens"),
        ("Prefix-cached tokens", "cached_tokens"),
        ("Preemptions", "preemptions"),
    ]:
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                number(
                    engines.get(n, {}).get("overall", {}).get("native", {}).get(key), 0
                )
                for n in ["gptq", "exl3"]
            )
            + " |"
        )
    for label, key in [
        ("Prefix-cache hit rate", "cached_fraction"),
        ("MTP accepted/drafted", "mtp_acceptance"),
    ]:
        vals = []
        for n in ["gptq", "exl3"]:
            value = engines.get(n, {}).get("overall", {}).get(key)
            vals.append("—" if value is None else number(100 * value) + "%")
        lines.append("| " + label + " | " + " | ".join(vals) + " |")
    lines += [
        "",
        "## Context bands",
        "",
        "Weighted native rates. Empty bands are unmeasured. Sparse bands and different model-generated content limit direct comparisons.",
        "",
        "| Input context | GPTQ requests | GPTQ prefill | GPTQ decode | EXL3 requests | EXL3 prefill | EXL3 decode |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    maximum = max(len(x["bands"]) for x in engines.values())
    for i in range(maximum):
        vals = []
        for name in ["gptq", "exl3"]:
            band = next(
                (
                    x
                    for x in engines.get(name, {}).get("bands", [])
                    if x["lower"] == i * 10000
                ),
                {},
            )
            vals += [
                str(band.get("requests", 0)),
                number(band.get("prefill_tps")),
                number(band.get("decode_tps")),
            ]
        lines.append(f"| {i * 10}–{(i + 1) * 10}K | " + " | ".join(vals) + " |")
    lines += [
        "",
        "## Functional outcomes",
        "",
    ]
    for name, engine in engines.items():
        q = engine["quality"]
        failed = [x["id"] for x in q["checks"] if not x["passed"]]
        lines.append(
            f"- **{name.upper()}**: {q['passed']}/{q['total']} functional cases passed. "
            + (
                "Failed cases: " + "; ".join(f"`{x}`" for x in failed) + "."
                if failed
                else "No failed final cases."
            )
        )
        if engine["session_status"] == "failed":
            lines.append(
                f"  The session stopped at its declared task budget: `{engine['stop_reason']}`. Its wall time is time to failure, not time to finish the task. Final output was graded unmodified after stopping with the same {q['total']} cases."
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "The task, starter, tools, sampling and acceptance cases are frozen. The agents can choose different edits, tests and answer lengths, so their histories and MTP acceptance need not match. End-to-end time and final functional score measure the useful outcome; native rates describe the requests actually generated. This single run does not establish a general code-quality or quantization ranking.",
        "",
        f"Prefill measures newly computed tokens with the session prefix cache enabled. It does not measure cold prefill of the entire growing context. Tool execution is excluded from native phase timing and included in end-to-end time. End-to-end time also includes the CPU tokenize preflight, metric polling and fixed independent grading between stages; startup and the initial warmup are excluded. Generated reasoning remains in the conversation, consistently on both sides. Each answer has a {engines['gptq']['provenance']['sampling']['thinking_token_budget']:,}-token thinking budget and at most 16,384 generated tokens, further bounded by a common 200,704-token input/output window.",
        "",
        "GPTQ uses the current production-onednn-v2 profile; EXL3 uses the existing 4.00-bpw checkpoint (6-bpw output head), vLLM 0.26.1 and its existing MTP3/pruned-vocabulary recipe. GPTQ uses MTP4/full draft vocabulary and vLLM 0.30.0. Both run exclusively on the same B70 at 180 W. No claim isolates INT4 versus EXL3 from these other differences.",
        "",
        "See comparison.json for exact image IDs, request provenance, accounting definitions, failed cases and project hashes. Generated projects, numeric request records, post-run acceptance results and compressed final model conversations are supplied alongside this report. Each saved measured request was checked against its prefix in that conversation; full repeated payloads and SSE remain local. Identical final-output grading is excluded from reported session elapsed time; supplemental-review-timing.json records a brief CPU-only GPTQ review that overlapped EXL3 requests 4–5; the fixed grading between stages remains included. A context-budget failure is reported as an incomplete task and is never ranked as a faster completion.",
    ]
    (a.output / "README.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                n: {
                    "requests": x["requests"],
                    "context_max": x["overall"]["context_max"],
                    "quality": {k: x["quality"][k] for k in ["passed", "total"]},
                }
                for n, x in engines.items()
            }
        )
    )


if __name__ == "__main__":
    main()

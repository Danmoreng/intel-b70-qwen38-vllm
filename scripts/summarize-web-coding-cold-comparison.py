#!/usr/bin/env python3
"""Audit matched cold prompts, separating preempted diagnostics from controls."""

import argparse
import gzip
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "runner", REPO / "scripts/run-web-coding-benchmark.py"
)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def load(p):
    return json.loads(p.read_text())


def validate(row, request, original):
    expected = dict(original)
    expected.update(
        model=request["model"],
        max_tokens=1024,
        ignore_eos=True,
        cache_salt="flappy-v4-common-prefix-20261001-" + str(row["source_request"]),
    )
    if request != expected:
        raise RuntimeError("common prompt differs")
    d = row["native"]
    u = row["usage"]
    preflight = row["tokenize_preflight"]
    if (
        d != R.delta(row["native_after"], row["native_before"])
        or d["completed"] != 1
        or d["generation_tokens"] != 1024
        or u["completion_tokens"] != 1024
        or d["cached_tokens"] != 0
        or d["prompt_tokens"] != u["prompt_tokens"]
        or d["prefill_tokens"] != u["prompt_tokens"]
        or preflight["prompt_tokens"] != u["prompt_tokens"]
        or preflight["max_output_tokens"] != 1024
        or d["preemptions"] < 0
    ):
        raise RuntimeError("cold throughput accounting differs")
    row["prefill_tps"] = d["prefill_tokens"] / d["prefill_seconds"]
    row["decode_tps"] = 1023 / d["decode_seconds"]
    row["classification"] = (
        "clean control" if d["preemptions"] == 0 else "preempted diagnostic"
    )
    return {k: v for k, v in request.items() if k != "model"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--retry", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    initial = load(args.initial / "replay.json")
    retry = load(args.retry / "replay.json")
    diagnostic = load(args.diagnostic / "diagnostic.json")
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = []
    for selected in initial["selected_requests"]:
        index = selected["request"]
        originalfile = args.initial / f"original-request-{index:04d}.json"
        if R.sha(originalfile) != selected["sha256"]:
            raise RuntimeError("original input hash differs")
        original = load(originalfile)
        g = next(
            dict(x)
            for x in initial["engines"]["gptq"]["requests"]
            if x["source_request"] == index
        )
        g.setdefault("identity", initial["engines"]["gptq"]["identity"])
        greq = args.initial / "gptq" / f"request-{index:04d}.json"
        if index == 47:
            e = next(
                dict(x)
                for x in initial["engines"]["exl3"]["requests"]
                if x["source_request"] == index
            )
            e.setdefault("identity", initial["engines"]["exl3"]["identity"])
            ereq = args.initial / "exl3" / f"request-{index:04d}.json"
        elif index == 65:
            e = load(args.retry / "exl3" / f"observation-{index:04d}.json")
            e["source_request"] = index
            ereq = args.retry / "exl3" / f"request-{index:04d}.json"
        else:
            if diagnostic["status"] != "complete":
                raise RuntimeError(
                    "188K diagnostic incomplete; do not publish as measured"
                )
            e = dict(diagnostic)
            ereq = args.diagnostic / "request.json"
        if validate(g, load(greq), original) != validate(e, load(ereq), original):
            raise RuntimeError("profile inputs differ")
        gresponse = args.initial / "gptq" / f"response-{index:04d}.json"
        eresponse = (
            (args.initial / "exl3" / f"response-{index:04d}.json")
            if index == 47
            else (
                (args.retry / "exl3" / f"response-{index:04d}.json")
                if index == 65
                else (args.diagnostic / "response.json")
            )
        )
        if (
            load(gresponse)["usage"] != g["usage"]
            or load(eresponse)["usage"] != e["usage"]
        ):
            raise RuntimeError("saved response usage differs")
        g["response_sha256"] = R.sha(gresponse)
        e["response_sha256"] = R.sha(eresponse)
        g["request_sha256"] = R.sha(greq)
        e["request_sha256"] = R.sha(ereq)
        if (
            g["identity"]["image_id"]
            != initial["engines"]["gptq"]["identity"]["image_id"]
            or e["identity"]["image_id"]
            != initial["engines"]["exl3"]["identity"]["image_id"]
        ):
            raise RuntimeError("worker image differs")
        pairs.append(
            {
                "source_request": index,
                "original_input": selected["original_context"],
                "original_sha256": selected["sha256"],
                "gptq": g,
                "exl3": e,
            }
        )
        (args.output / f"cold-original-request-{index:04d}.json.gz").write_bytes(
            gzip.compress(originalfile.read_bytes(), mtime=0)
        )
    result = {
        "scope": "Identical archived v4 coding histories; cold KV; 1024 outputs; no tool execution or quality grade. Report preempted requests as diagnostics, not zero-preemption controls. One observation per context, whole serving profiles.",
        "pairs": pairs,
        "initial_attempt_status": initial["status"],
        "initial_attempt_error": initial.get("error"),
        "retry_status": retry["status"],
        "retry_error": retry.get("error"),
        "diagnostic_status": diagnostic["status"],
        "sources": {
            str(p): R.sha(p)
            for p in [
                args.initial / "replay-source.py",
                args.retry / "replay-source.py",
                args.diagnostic / "diagnostic-source.py",
            ]
        },
    }
    R.save(args.output / "cold-comparison.json", result)
    lines = [
        "# Matched coding histories: cold throughput and preemption diagnostics",
        "",
        "The original strict control required zero preemptions. EXL3 failed that criterion at 139K, also on a fresh worker. Its exact-count, zero-cache observations are reported explicitly as preempted diagnostics; no failed measurement is silently relabelled as a clean control. GPTQ points have zero preemptions. All six inputs, native counter differences, output counts and model identities were audited.",
        "",
        "| Actual input | GPTQ cold prefill | EXL3 cold prefill | GPTQ decode | EXL3 decode | Preemptions GPTQ / EXL3 | Classification |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for pair in pairs:
        g = pair["gptq"]
        e = pair["exl3"]
        assert g["usage"]["prompt_tokens"] == e["usage"]["prompt_tokens"]
        lines.append(
            f"| {g['usage']['prompt_tokens']:,} | {g['prefill_tps']:,.1f} | {e['prefill_tps']:,.1f} | {g['decode_tps']:,.1f} | {e['decode_tps']:,.1f} | {g['native']['preemptions']:.0f} / {e['native']['preemptions']:.0f} | {'Clean control' if e['native']['preemptions'] == 0 else 'Preempted diagnostic'} |"
        )
    lines += [
        "",
        "Rates are tokens per native phase second. Prefix-cached tokens are zero; all input tokens are computed, and exactly 1,024 outputs are generated with ignored EOS. Decode counts the 1,023 tokens after the first. Model ID is the only profile-specific payload field. Sampling and seeds match the original archived request; generated continuations may differ. Outputs are truncated and tool calls are not executed.",
        "",
        "The EXL3 139K and 188K requests use separate fresh workers after excluded warmup. The GPTQ three-point series and EXL3 103K point come from the initial fresh-worker control. Cold token accounting is verified for every observation. These probes run after game reviews, without simultaneous browser testing. They compare complete pinned serving recipes, not isolated quantization or kernels. The primary adaptive v6 sessions had zero preemptions throughout.",
        "",
        "See cold-comparison.json for native before/after counters, worker identities, exact hashes and the rejected zero-preemption attempt. The compressed original prompts make the common input reviewable.",
    ]
    (args.output / "COLD_COMPARISON.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[4:]))


if __name__ == "__main__":
    main()

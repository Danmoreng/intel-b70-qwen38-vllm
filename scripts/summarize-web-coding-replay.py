#!/usr/bin/env python3
"""Verify identical cold coding prompts and export a small throughput control."""

import argparse
import gzip
import importlib.util
import json
import math
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
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads((args.root / "replay.json").read_text())
    if data["status"] != "complete" or set(data["engines"]) != {"gptq", "exl3"}:
        raise RuntimeError("control did not complete on both profiles")
    args.output.mkdir(parents=True, exist_ok=True)
    for selected in data["selected_requests"]:
        index = selected["request"]
        file = args.root / f"original-request-{index:04d}.json"
        if R.sha(file) != selected["sha256"]:
            raise RuntimeError("original prompt hash differs")
        original = json.loads(file.read_text())
        payloads = []
        for name in ["gptq", "exl3"]:
            rows = data["engines"][name]["requests"]
            row = next(x for x in rows if x["source_request"] == index)
            request_file = args.root / name / f"request-{index:04d}.json"
            request = json.loads(request_file.read_text())
            expected = dict(original)
            expected.update(
                model=request["model"],
                max_tokens=1024,
                ignore_eos=True,
                cache_salt="flappy-v4-common-prefix-20261001-" + str(index),
            )
            if request != expected:
                raise RuntimeError(
                    "control request differs from declared common settings"
                )
            payloads.append({k: v for k, v in request.items() if k != "model"})
            native = row["native"]
            if native != R.delta(row["native_after"], row["native_before"]):
                raise RuntimeError("native counters differ")
            if (
                native["completed"] != 1
                or native["generation_tokens"] != 1024
                or native["cached_tokens"] != 0
                or native["preemptions"] != 0
                or native["prefill_tokens"] != row["usage"]["prompt_tokens"]
                or row["tokenize_preflight"]["prompt_tokens"]
                != row["usage"]["prompt_tokens"]
            ):
                raise RuntimeError("control is not cold fixed-length throughput")
            for key, value in [
                ("prefill_tps", native["prefill_tokens"] / native["prefill_seconds"]),
                ("decode_tps", 1023 / native["decode_seconds"]),
            ]:
                if not math.isclose(value, row[key], rel_tol=1e-12):
                    raise RuntimeError("rate differs")
            row["request_sha256"] = R.sha(request_file)
            row["response_sha256"] = R.sha(
                args.root / name / f"response-{index:04d}.json"
            )
        if payloads[0] != payloads[1]:
            raise RuntimeError("the two profiles received different control inputs")
        (args.output / (f"replay-original-request-{index:04d}.json.gz")).write_bytes(
            gzip.compress(file.read_bytes(), mtime=0)
        )
    data["replay_source_sha256"] = R.sha(args.root / "replay-source.py")
    R.save(args.output / "controlled-replay.json", data)
    lines = [
        "# Identical recorded coding prompts: cold KV control",
        "",
        "Three archived WebGL coding histories from the initial v4 run, reused byte-for-byte on both complete serving profiles. Each has its own cache salt; zero cached tokens, exact prompt-token accounting and zero preemptions are verified. A fresh worker and excluded warmup precede each profile.",
        "",
        "| Source request / input | GPTQ actual input | EXL3 actual input | GPTQ cold prefill | EXL3 cold prefill | GPTQ decode | EXL3 decode |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for selected in data["selected_requests"]:
        index = selected["request"]
        g = next(
            x
            for x in data["engines"]["gptq"]["requests"]
            if x["source_request"] == index
        )
        e = next(
            x
            for x in data["engines"]["exl3"]["requests"]
            if x["source_request"] == index
        )
        lines.append(
            f"| {index} / {selected['original_context']:,} | {g['usage']['prompt_tokens']:,} | {e['usage']['prompt_tokens']:,} | {g['prefill_tps']:,.1f} | {e['prefill_tps']:,.1f} | {g['decode_tps']:,.1f} | {e['decode_tps']:,.1f} |"
        )
    lines += [
        "",
        "Rates are native tokens per phase second. Prefill is the full uncached input; decode counts 1,023 tokens after the first from exactly 1,024 generated tokens. `ignore_eos=true` supplies a fixed throughput window; model tool calls are recorded but not executed. These truncated outputs have no code-quality score. Other sampling, thinking and tool settings match the archived request. Model ID is the only profile-specific request field.",
        "",
        "This is one observation at each of three contexts, not a distribution or a continuous agent run. It complements the adaptive coding session, whose histories differ. It compares the complete GPTQ/MTP4/full-vocabulary and EXL3/MTP3/pruned-vocabulary recipes, rather than isolating quantization or any one kernel. Exact native counters, identities, selected hashes and compressed original requests accompany the report.",
    ]
    (args.output / "CONTROLLED_REPLAY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[4:9]))


if __name__ == "__main__":
    main()

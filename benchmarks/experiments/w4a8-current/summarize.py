#!/usr/bin/env python3
"""Compare the new W4A8 rows with archived, prompt-matched GPTQ/EXL3 rows."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent / "runs/2026-09-29/w4a8-180w.jsonl"
GPTQ = ROOT / "benchmark-results/meaningful-full-profile/run-20260923-201101-w0.00"
EXL3 = ROOT.parent / "exl3xpu-evaluation/results/exl3xpu-20260928.jsonl"
CASES = (("phase-4k-c1", 3), ("phase-32k-c1", 3),
         ("phase-128k-c1", 3))


def median(rows, key):
    return statistics.median(row[key] for row in rows)


def main() -> None:
    w4a8 = {(d["case"], d["repeat"]): d for line in RUN.read_text().splitlines()
            if (d := json.loads(line))}
    exl3 = {(d["case"], d["repeat"]): d for line in EXL3.read_text().splitlines()
            if (d := json.loads(line))}
    print("case n GPTQ_prefill W4A8_prefill EXL3_prefill GPTQ_decode W4A8_decode EXL3_decode W4A8_TTFT W4A8_wall preemptions")
    for case, n in CASES:
        paired = [(case, i) for i in range(1, n + 1)]
        if any(key not in w4a8 or key not in exl3 for key in paired):
            raise RuntimeError(f"missing rows for {case}")
        w = [w4a8[key] for key in paired]
        e = [exl3[key] for key in paired]
        g = [json.loads((GPTQ / f"{case}-r{i}" / "summary.json").read_text())
             for _, i in paired]
        for wr, er, gr in zip(w, e, g):
            if wr["prompt_sha256"] != er["prompt_sha256"]:
                raise RuntimeError(f"prompt hash mismatch: {case}")
            if wr["prompt_tokens"] != gr["requested_prompt_tokens"]:
                raise RuntimeError(f"prompt token mismatch: {case}")
            if wr["power_cap_w"] != 180 or wr["completion_tokens"] != 1024:
                raise RuntimeError(f"invalid power cap or completion count: {case}")
            if wr["prompt_cached_tokens"] != 0:
                raise RuntimeError(f"cached prefill: {case}")
        values = [
            case, n,
            median(g, "native_prefill_compute_tokens_per_s"),
            median(w, "native_prefill_tps"),
            median(e, "native_prefill_tps"),
            median(g, "native_weighted_decode_tokens_per_s"),
            median(w, "native_decode_tps"),
            median(e, "native_decode_tps"),
            median(w, "ttft_s"), median(w, "wall_s"),
            sum(row["native_metric_deltas"].get("vllm:num_preemptions_total", 0)
                for row in w),
        ]
        print(values[0], values[1], *(f"{v:.2f}" for v in values[2:]))


if __name__ == "__main__":
    main()

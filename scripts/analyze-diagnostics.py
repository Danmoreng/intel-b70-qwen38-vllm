#!/usr/bin/env python3
"""Analyze a pre-capture XPU trace and emit decode-only cost attribution."""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path


def in_windows(event: dict, windows: list[tuple[float, float]]) -> bool:
    start = event.get("ts", -1)
    stop = start + event.get("dur", 0)
    return any(left <= start and stop <= right + 1 for left, right in windows)


def group(kernel: str, parent: str) -> str:
    lower = kernel.lower()
    if parent == "_xpu_C::int4_gemm_w4a16":
        return "INT4 W4A16 GEMM"
    if "fmha" in lower or "flash_attn" in lower or "paged_attention" in lower:
        return "attention"
    if kernel.startswith("gdn::") or "gated_delta" in lower:
        return "GDN"
    if any(token in lower for token in ("sample", "argmax", "topk", "softmax")):
        return "selection/softmax"
    if "gemm" in lower or parent in {"aten::mm", "aten::linear", "aten::matmul"}:
        return "other GEMM"
    if "memcpy" in lower or "copy" in lower:
        return "copies"
    return "other kernels"


def union_duration(spans: list[tuple[float, float]]) -> float:
    active = 0.0
    right = None
    for left, stop in sorted(spans):
        if right is None or left > right:
            active += stop - left
        elif stop > right:
            active += stop - right
        right = max(right or stop, stop)
    return active


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    traces = sorted((args.run_dir / "decode-trace" / "traces").rglob("*.pt.trace.json.gz"))
    if len(traces) != 1:
        raise RuntimeError(f"expected exactly one trace, found {len(traces)}")
    with gzip.open(traces[0], "rt") as source:
        events = json.load(source)["traceEvents"]

    annotations = [event for event in events if event.get("cat") == "user_annotation"]
    decode_annotations = [
        event
        for event in annotations
        if "execute_" in event.get("name", "")
        and "_context_0" in event.get("name", "")
        and "_generation_1" in event.get("name", "")
        and event.get("ph") == "X"
    ]
    windows = [
        (event["ts"], event["ts"] + event.get("dur", 0)) for event in decode_annotations
    ]
    if not windows:
        raise RuntimeError("no decode-only execute annotations found")

    cpu_by_external = {
        event.get("args", {}).get("External id"): event
        for event in events
        if event.get("cat") == "cpu_op"
    }
    kernels = [
        event
        for event in events
        if event.get("cat") == "kernel" and event.get("ph") == "X" and in_windows(event, windows)
    ]
    groups = collections.Counter()
    names = collections.Counter()
    parents = collections.Counter()
    shapes = collections.Counter()
    spans = []
    for event in kernels:
        parent_event = cpu_by_external.get(event.get("args", {}).get("External id"), {})
        parent = parent_event.get("name", "unmapped")
        seconds = event.get("dur", 0) / 1e6
        groups[group(event.get("name", ""), parent)] += seconds
        names[event.get("name", "")] += seconds
        parents[parent] += seconds
        input_dims = parent_event.get("args", {}).get("Input Dims")
        shapes[f"{parent} {input_dims}"] += seconds
        spans.append((event["ts"], event["ts"] + event.get("dur", 0)))

    replay_names = {
        "zeCommandListImmediateAppendCommandListsExp",
        "zeCommandListAppendLaunchKernel",
    }
    replay_events = [
        event
        for event in events
        if event.get("name") in replay_names and in_windows(event, windows)
    ]
    total = sum(groups.values())
    window_s = sum(right - left for left, right in windows) / 1e6
    graph_internals_visible = (
        len(kernels) >= 100 * len(windows)
        and groups["INT4 W4A16 GEMM"] > 0
        and groups["GDN"] > 0
    )
    result = {
        "trace": str(traces[0]),
        "trace_size_bytes": traces[0].stat().st_size,
        "decode_iterations": len(windows),
        "decode_annotation_examples": [event["name"] for event in decode_annotations[:5]],
        "decode_window_s": window_s,
        "visible_decode_kernel_events": len(kernels),
        "visible_decode_kernel_sum_s": total,
        "visible_decode_kernel_union_s": union_duration(spans) / 1e6,
        "graph_runtime_events_in_decode": len(replay_events),
        "graph_replay_internals_visible": graph_internals_visible,
        "graph_visibility_gate": (
            "at least 100 kernels per decode iteration and visible INT4 GEMM plus GDN"
        ),
        "groups_s": dict(groups),
        "top_kernel_s": dict(names.most_common(30)),
        "top_cpu_parent_s": dict(parents.most_common(30)),
        "top_operator_shapes_s": dict(shapes.most_common(30)),
        "scope": "kernels fully contained in context=0,generation=1 execute annotations",
        "warning": "Profiler overhead makes this diagnostic attribution, not a throughput score.",
    }
    (args.run_dir / "decode-trace-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# M01 full decode trace",
        "",
        "This is diagnostic attribution, not a throughput score. The profiler was active before XPU graph capture.",
        "",
        f"- Decode iterations: {len(windows)}",
        f"- Visible decode kernel events: {len(kernels)}",
        f"- Decode annotation wall span (sum): {window_s:.3f} s",
        f"- Visible GPU kernel union: {result['visible_decode_kernel_union_s']:.3f} s",
        f"- Graph replay internals visible: {result['graph_replay_internals_visible']}",
        "",
        "## Visible decode kernel groups",
        "",
        "| Group | Kernel time (s) | Share |",
        "|---|---:|---:|",
    ]
    for name, seconds in groups.most_common():
        share = 100 * seconds / total if total else 0
        lines.append(f"| {name} | {seconds:.4f} | {share:.1f}% |")
    (args.run_dir / "decode-trace-summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

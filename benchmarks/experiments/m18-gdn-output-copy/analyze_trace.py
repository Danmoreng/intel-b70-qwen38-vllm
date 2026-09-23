#!/usr/bin/env python3
"""Classify standalone FP16 copies in a vLLM GDN serving trace."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.trace, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    kernels = sorted(
        (event for event in events if event.get("cat") == "kernel"),
        key=lambda event: event["ts"],
    )
    gdn_positions = [
        index for index, event in enumerate(kernels)
        if "gdn::gated_delta_rule" in event["name"]
    ]
    next_names = Counter(
        tuple(event["name"] for event in kernels[index + 1:index + 4])
        for index in gdn_positions
    )
    standalone_copy = lambda event: (
        "CopyScalarFunc" in event["name"] or "Memcpy" in event["name"]
    )
    following_copies = sum(
        any(standalone_copy(event) for event in kernels[index + 1:index + 4])
        for index in gdn_positions
    )
    cpu_copies = {
        event.get("args", {}).get("External id"): event
        for event in events
        if event.get("cat") == "cpu_op" and event.get("name") == "aten::copy_"
    }
    copy_shapes = Counter(
        str(cpu_copies.get(event.get("args", {}).get("External id"), {})
            .get("args", {}).get("Input Dims"))
        for event in kernels
        if "CopyScalarFunc<c10::Half>" in event["name"]
    )
    result = {
        "trace": str(args.trace),
        "trace_events": len(events),
        "device_kernels": len(kernels),
        "gdn_gated_delta_rule_count": len(gdn_positions),
        "standalone_copy_within_next_3_kernels_after_gdn_rule": following_copies,
        "next_3_kernel_sequences": [
            {"count": count, "names": list(names)}
            for names, count in next_names.most_common()
        ],
        "standalone_fp16_copy_shapes": dict(copy_shapes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "gdn_gated_delta_rule_count",
        "standalone_copy_within_next_3_kernels_after_gdn_rule",
        "standalone_fp16_copy_shapes",
    )}, indent=2))


if __name__ == "__main__":
    main()

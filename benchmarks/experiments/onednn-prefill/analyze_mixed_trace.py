#!/usr/bin/env python3
"""Check the captured pinned-vLLM mixed attention metadata contract."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "mixed-trace-4k-32k-routes.txt"


def main():
    traces = []
    for line in SOURCE.read_text().splitlines():
        marker = "B70_ONEDNN_MIXED_TRACE "
        if marker not in line:
            continue
        row = ast.literal_eval(line.split(marker, 1)[1])
        total_q, max_kv, cu_count = row["signature"]
        cu = row["cu_seqlens_q"]
        used = row["seqused_k"]
        block_rows, _ = row["block_table_shape"]
        if not (cu[0] == 0 and cu[-1] == total_q and
                len(cu) == cu_count == len(used) + 1 and
                len(used) == block_rows and
                all(left <= right for left, right in zip(cu, cu[1:]))):
            raise ValueError(f"inconsistent mixed metadata: {row}")
        query_lengths = [right - left for left, right in zip(cu, cu[1:])]
        if any(q > kv for q, kv in zip(query_lengths, used)):
            raise ValueError(f"query longer than active KV: {row}")
        traces.append({
            "total_q": total_q, "max_seqlen_k": max_kv,
            "query_lengths": query_lengths, "active_kv_lengths": used,
            "block_table_shape": row["block_table_shape"],
            "block_table_stride": row["block_table_stride"],
            "q_stride": row["q_stride"],
            "scheduler_metadata": row["scheduler_metadata"],
        })

    real = [row for row in traces if row["total_q"] >= 256 and
            max(row["active_kv_lengths"]) >= 1024]
    if not real:
        raise ValueError("no real mixed serving shapes captured")
    eligible = []
    for row in real:
        for index, (query, length) in enumerate(zip(
                row["query_lengths"], row["active_kv_lengths"])):
            if query >= 256 and 16384 <= length <= 196608:
                eligible.append({"total_q": row["total_q"], "request_index": index,
                                 "query_rows": query, "active_kv": length})
    result = {
        "source": SOURCE.name,
        "trace_image_id": json.loads((ROOT / "mixed-trace-4k-32k.json").read_text())["image_id"],
        "all_signatures": len(traces), "real_mixed_signatures": len(real),
        "captured_real_shapes": real,
        "potential_single_request_onednn_subcalls": eligible,
        "caveat": "The trace proves shape/layout facts for this diagnostic C2 run only. It does not prove a split implementation numerically correct or faster.",
    }
    output = ROOT / "mixed-metadata-contract-4k-32k.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"all_signatures": len(traces),
                      "real_mixed_signatures": len(real),
                      "potential_onednn_subcalls": len(eligible)}), flush=True)


if __name__ == "__main__":
    main()

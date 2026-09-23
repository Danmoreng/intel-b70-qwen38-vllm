#!/usr/bin/env python3
"""Check whether the GDN output reshapes dispatch an XPU copy kernel."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNormGated


def event_time(fn):
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def trace(fn, path):
    for _ in range(5):
        fn()
    torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        for _ in range(5):
            fn()
        torch.xpu.synchronize()
    prof.export_chrome_trace(str(path))
    raw = json.loads(path.read_text())
    events = raw["traceEvents"]
    kernels = [row for row in events if row.get("cat") == "kernel"]
    copies = [row for row in kernels if any(word in row.get("name", "").lower()
              for word in ("copy", "reshape", "contiguous"))]
    return {
        "kernel_count": len(kernels),
        "kernel_names": sorted(set(row["name"] for row in kernels)),
        "copy_like_kernels": [{"name": row["name"], "duration_us": row.get("dur")}
                              for row in copies],
        "trace_file": str(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(38)
    tokens, heads, dim, hidden = 4, 48, 128, 5120
    core = torch.randn((tokens, heads, dim), device="xpu", dtype=torch.float16)
    gate = torch.randn_like(core)
    with set_current_vllm_config(VllmConfig()):
        norm = RMSNormGated(
            dim, eps=1e-6, group_size=None, norm_before_gate=True,
            activation="silu", device=torch.device("xpu"), dtype=torch.float16,
        )
    projection = torch.nn.Linear(heads * dim, hidden, bias=False,
                                 device="xpu", dtype=torch.float16)

    def current():
        shape = gate.shape
        x = norm(core.reshape(-1, dim), gate.reshape(-1, dim))
        x = x.reshape(shape).flatten(-2)
        return projection(x)

    def proposed():
        return projection(norm(core, gate).flatten(-2))

    a = current()
    b = proposed()
    torch.xpu.synchronize()
    if not torch.allclose(a, b, rtol=0.01, atol=0.002):
        raise RuntimeError("output differs")
    rows = {}
    for mode in ("eager", "compiled"):
        if mode == "compiled":
            try:
                functions = {
                    "current": torch.compile(current, fullgraph=True),
                    "proposed": torch.compile(proposed, fullgraph=True),
                }
                for fn in functions.values():
                    fn()
            except Exception as error:
                rows[mode] = {"error": repr(error)}
                continue
        else:
            functions = {"current": current, "proposed": proposed}
        for fn in functions.values():
            for _ in range(5):
                fn()
        samples = {name: [] for name in functions}
        for i in range(16):
            for name in (("current", "proposed") if i % 2 == 0 else
                         ("proposed", "current")):
                samples[name].append(event_time(functions[name]))
        traces = {}
        for name, fn in functions.items():
            traces[name] = trace(fn, args.output.with_name(
                args.output.stem + f"-{mode}-{name}.trace.json"
            ))
        rows[mode] = {
            "samples_ms": samples,
            "median_ms": {name: statistics.median(values)
                          for name, values in samples.items()},
            "traces": traces,
        }
        print(json.dumps({"mode": mode, "median_ms": rows[mode]["median_ms"],
                          "copy_like_counts": {name: len(value["copy_like_kernels"])
                                               for name, value in traces.items()}}), flush=True)
    result = {
        "scope": "synthetic GDN output region with production dimensions and dense stand-in projection",
        "shape": {"tokens": tokens, "heads": heads, "head_dim": dim,
                  "hidden_size": hidden},
        "torch_version": torch.__version__,
        "output_allclose": True,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()

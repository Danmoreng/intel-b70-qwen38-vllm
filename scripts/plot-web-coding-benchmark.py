#!/usr/bin/env python3
"""Plot validated request-weighted native rates, preserving unmeasured bands."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison", type=Path)
    args = parser.parse_args()
    result = json.loads(args.comparison.read_text())
    profiles = {
        "gptq": ("GPTQ production v2", "#147d92"),
        "exl3": ("EXL3 4.00 bpw", "#b85425"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for axis, key, label in zip(
        axes,
        ["prefill_tps", "decode_tps"],
        ["Native prefill of new KV tokens", "Native decode after first token"],
        strict=True,
    ):
        for engine, data in result["engines"].items():
            name, color = profiles[engine]
            bands = data["bands"]
            xs = [(b["lower"] + b["upper"]) / 2000 for b in bands]
            ys = [b[key] if b[key] is not None else float("nan") for b in bands]
            axis.plot(
                xs, ys, "o-", label=name, color=color, markersize=4, linewidth=1.5
            )
        axis.set_title(label, fontsize=11)
        axis.set_xlabel("Rendered input context (K tokens; band midpoint)")
        axis.set_ylabel("Tokens / native phase second")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Flappy Bird "
        + result["engines"]["gptq"]["fixture_id"].removeprefix("flappybird-webgl2-")
        + ": one adaptive coding run per serving profile",
        fontsize=13,
    )
    fig.supxlabel(
        "Weighted 10K bands; retained reasoning; prefix caching enabled. Histories differ; empty bands remain unmeasured.",
        fontsize=8,
    )
    for extension in ["png", "svg"]:
        fig.savefig(args.comparison.parent / ("context-rates." + extension), dpi=160)
    svg = args.comparison.parent / "context-rates.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()

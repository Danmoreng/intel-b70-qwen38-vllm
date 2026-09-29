#!/usr/bin/env python3
"""Freeze twelve new source contexts for the final C2 quality screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

from meaningful_benchmark import CHAT_TEMPLATE_KWARGS, CORPUS, load_corpus, make_prompt


MODEL = "Qwen3.8-27B"
TARGETS = (4096, 4096, 32768, 32768, 32768, 32768, 32768, 32768,
           131072, 131072, 199000, 199000)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--container", default="b70-qwen38-vllm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists; frozen contexts are immutable")
    image_id = subprocess.check_output(
        ["docker", "inspect", args.container, "--format", "{{.Image}}"],
        text=True).strip()
    sources, corpus_hash = load_corpus(CORPUS)
    counts: dict[str, int] = {}

    def count(prompt: str) -> int:
        key = digest(prompt.encode())
        if key not in counts:
            body = {"model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "chat_template_kwargs": CHAT_TEMPLATE_KWARGS}
            request = urllib.request.Request(
                args.base + "/tokenize", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=180) as response:
                counts[key] = int(json.load(response)["count"])
        return counts[key]

    args.output.mkdir(parents=True)
    rows = []
    for index, target in enumerate(TARGETS, 1):
        case_id = f"b70-release-heldout-v1-{index:02d}-{corpus_hash[:12]}"
        full_prompt, _, paths = make_prompt(sources, target, case_id, count)
        context, question = full_prompt.rsplit("\n\n", 1)
        if not question or not context.endswith("\n"):
            raise RuntimeError(f"invalid generated source context {case_id}")
        actual = count(context)
        filename = f"context-{index:02d}.txt"
        data = context.encode()
        (args.output / filename).write_bytes(data)
        row = {"id": f"context-{index:02d}", "target_tokens": target,
               "calibrated_context_tokens": actual,
               "source_paths": paths, "source_context_sha256": digest(data),
               "filename": filename}
        rows.append(row)
        print(json.dumps({"id": row["id"], "target": target,
                          "tokens": actual, "files": len(paths)}), flush=True)
    if len({row["source_context_sha256"] for row in rows}) != 12:
        raise RuntimeError("held-out contexts are not distinct")
    manifest = {"schema": 1, "purpose": "B70 C2 held-out quality qualification",
                "model": MODEL, "image_id": image_id,
                "corpus_sha256": corpus_hash,
                "generator_sha256": digest(Path(__file__).read_bytes()),
                "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
                "contexts": rows}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"state": "complete", "output": str(args.output),
                      "manifest_sha256": digest(
                          (args.output / "manifest.json").read_bytes())}),
          flush=True)


if __name__ == "__main__":
    main()

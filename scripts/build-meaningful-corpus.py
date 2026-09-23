#!/usr/bin/env python3
"""Freeze public repository source and documentation for reproducible benchmarks."""

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "benchmarks" / "meaningful-corpus.json"
SUFFIXES = {".py", ".md", ".js", ".sh", ".cpp", ".patch"}


def main() -> None:
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    sources = []
    for name in sorted(filter(None, names)):
        path = ROOT / name
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        if name.startswith("benchmarks/runs/") and path.suffix != ".md":
            continue
        content = path.read_text(errors="replace")
        if len(content.strip()) < 160:
            continue
        sources.append({
            "path": name,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "content": content,
        })
    if not sources:
        raise RuntimeError("no meaningful repository sources found")
    OUTPUT.write_text(json.dumps({
        "schema": 1,
        "provenance": "Frozen public source and documentation from this repository; regenerate explicitly with scripts/build-meaningful-corpus.py",
        "sources": sources,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"Frozen {len(sources)} sources ({OUTPUT.stat().st_size} bytes) in {OUTPUT}")


if __name__ == "__main__":
    main()

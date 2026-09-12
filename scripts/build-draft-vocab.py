#!/usr/bin/env python3
"""Build a reduced draft vocabulary from a user-supplied text/code corpus."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".css", ".go", ".h", ".hpp", ".html",
    ".java", ".js", ".json", ".jsonl", ".md", ".py", ".rs", ".sh",
    ".sql", ".toml", ".ts", ".tsx", ".txt", ".yaml", ".yml",
}


def input_files(inputs: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for item in inputs:
        if item.is_file():
            files.add(item.resolve())
        elif item.is_dir():
            files.update(
                path.resolve()
                for path in item.rglob("*")
                if path.is_file()
                and path.suffix.lower() in TEXT_SUFFIXES
                and ".git" not in path.parts
            )
        else:
            raise FileNotFoundError(item)
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Corpus files or directories")
    parser.add_argument("--model", required=True, help="Local model directory or HF model ID")
    parser.add_argument("--revision", help="Pinned HF revision when --model is an ID")
    parser.add_argument("--size", type=int, default=40960)
    parser.add_argument("--output", type=Path, default=Path("draft-vocab-40960.txt"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=True,
    )
    files = input_files(args.inputs)
    if not files:
        raise RuntimeError("No supported corpus files found")
    if args.size <= 0 or args.size > len(tokenizer):
        raise ValueError(f"--size must be between 1 and {len(tokenizer)}")
    if args.size % 8:
        raise ValueError("--size must be divisible by 8")

    counts: collections.Counter[int] = collections.Counter()
    bytes_read = 0
    for path in files:
        raw = path.read_bytes()
        bytes_read += len(raw)
        text = raw.decode(errors="replace")
        counts.update(tokenizer.encode(text, add_special_tokens=False))

    selected: list[int] = []
    seen: set[int] = set()
    special_ids = set(tokenizer.all_special_ids)
    special_ids.update(tokenizer.added_tokens_decoder)
    for token_id in sorted(special_ids):
        if 0 <= token_id < len(tokenizer) and token_id not in seen:
            seen.add(token_id)
            selected.append(token_id)
    for token_id, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if len(selected) == args.size:
            break
        if token_id not in seen:
            seen.add(token_id)
            selected.append(token_id)
    # Deterministic fallback for small corpora.
    for token_id in range(len(tokenizer)):
        if len(selected) == args.size:
            break
        if token_id not in seen:
            seen.add(token_id)
            selected.append(token_id)

    selected.sort()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(map(str, selected)) + "\n"
    args.output.write_text(payload)
    total = counts.total()
    covered = sum(count for token, count in counts.items() if token in seen)
    manifest = {
        "model": args.model,
        "revision": args.revision,
        "vocabulary_size": len(tokenizer),
        "draft_vocabulary_size": len(selected),
        "corpus_files": len(files),
        "corpus_bytes": bytes_read,
        "corpus_tokens": total,
        "corpus_coverage": covered / total if total else 0.0,
        "output": str(args.output),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    manifest_path = args.output.with_name("draft-vocab-manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()


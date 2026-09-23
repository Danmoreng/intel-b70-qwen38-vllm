#!/usr/bin/env python3
"""Freeze source-review prompts under token budgets for the context harness.

The historical filename is retained for callers. The recorded calibrated token
count is the actual count; meaningful source lines are never padded with noise.
"""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from meaningful_benchmark import CHAT_TEMPLATE_KWARGS, CORPUS, load_corpus, make_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--per-target", type=int, default=5)
    parser.add_argument("--system-prompt-file")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=True)
    system_prompt = None
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text().rstrip("\n")
    sources, corpus_sha256 = load_corpus(CORPUS)

    def messages_for(content: str) -> list[dict]:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def count(content: str) -> int:
        encoded = tokenizer.apply_chat_template(
            messages_for(content), tokenize=True, add_generation_prompt=True,
            return_dict=True, **CHAT_TEMPLATE_KWARGS,
        )
        return len(encoded["input_ids"])

    prompts = []
    for target in [int(value) for value in args.targets.split(",")]:
        for index in range(args.per_target):
            case_id = f"source-{target}-{index + 1}-{corpus_sha256[:12]}"
            content, actual, paths = make_prompt(sources, target, case_id, count)
            prompts.append({
                "target_tokens": target,
                "calibrated_tokens": actual,
                "family": "source-review",
                "scenario": "frozen_meaningful_source_context",
                "entropy_prefix": case_id,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "source_paths": paths,
                "messages": messages_for(content),
            })
    Path(args.output).write_text(json.dumps({
        "schema": 2,
        "tokenizer": args.model,
        "tokenizer_revision": args.revision,
        "corpus_sha256": corpus_sha256,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest()
        if system_prompt else None,
        "prompts": prompts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

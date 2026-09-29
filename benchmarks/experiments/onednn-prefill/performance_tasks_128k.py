"""Twelve paired tasks from two frozen 128K source-review contexts."""

import hashlib
import json

from performance_tasks import CODE, FROZEN, REVIEW, STRUCTURED, retrievals


def tasks():
    result = []
    for context_index in range(2):
        source = FROZEN / f"phase-128k-c1-r{context_index + 1}" / "prompt-1.txt"
        prompt = source.read_text()
        base, ending = prompt.rsplit("\n\n", 1)
        if not any(ending.startswith(prefix) for prefix in
                   ("Review the supplied", "Explain the purpose",
                    "Write a focused", "Write a detailed")):
            raise ValueError(f"unexpected frozen 128K prompt ending: {source}")
        base_sha = hashlib.sha256(base.encode()).hexdigest()
        prefix = f"context-128k-{context_index + 1}"
        for slot in range(2):
            specification, cases = CODE[2 * context_index + slot]
            result.append({"id": f"{prefix}-code-{slot + 1}", "kind": "code",
                           "context_sha256": base_sha, "cases": cases,
                           "prompt": base + "\n\nTASK: " + specification +
                           " Use only the Python standard library. Return only Python code defining solve.\n"})
        for slot, (path, phrase) in enumerate(retrievals(base), 1):
            result.append({"id": f"{prefix}-retrieval-{slot}", "kind": "retrieval",
                           "context_sha256": base_sha, "expected": path,
                           "prompt": base + "\n\nTASK: In which supplied FILE does this exact line occur? "
                           + json.dumps(phrase) + " Return only its path, with no explanation.\n"})
        snippet, issue, line = REVIEW[context_index]
        result.append({"id": f"{prefix}-review", "kind": "review",
                       "context_sha256": base_sha,
                       "expected": {"issue": issue, "line": line},
                       "prompt": base + "\n\nTASK: Review this separate Python snippet. "
                       "Return only JSON with keys issue and line. The issue must be one of "
                       "zero_division, mutable_default, off_by_one, missing_device. "
                       "Line is the one-based line number causing the bug.\n" + snippet})
        specification, expected = STRUCTURED[context_index]
        result.append({"id": f"{prefix}-structured", "kind": "structured",
                       "context_sha256": base_sha, "expected": expected,
                       "prompt": base + "\n\nTASK: " + specification +
                       " Return only a JSON object, with no explanation.\n"})
    if len(result) != 12 or len({row["id"] for row in result}) != 12:
        raise AssertionError("the 128K screen must contain 12 distinct tasks")
    return result

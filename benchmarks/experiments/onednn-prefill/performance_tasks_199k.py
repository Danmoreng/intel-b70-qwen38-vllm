"""Six paired tasks from the frozen near-maximum-context source prompt."""

import hashlib
import json

from performance_tasks import CODE, FROZEN, REVIEW, STRUCTURED, retrievals


def tasks():
    source = FROZEN / "full-context-199680-r1" / "prompt-1.txt"
    prompt = source.read_text()
    base, ending = prompt.rsplit("\n\n", 1)
    if not ending.startswith("Explain the purpose and behavior"):
        raise ValueError(f"unexpected frozen 199K prompt ending: {source}")
    base_sha = hashlib.sha256(base.encode()).hexdigest()
    result = []
    for slot in range(2):
        specification, cases = CODE[4 + slot]
        result.append({"id": f"context-199k-code-{slot + 1}", "kind": "code",
                       "context_sha256": base_sha, "cases": cases,
                       "prompt": base + "\n\nTASK: " + specification +
                       " Use only the Python standard library. Return only Python code defining solve.\n"})
    for slot, (path, phrase) in enumerate(retrievals(base), 1):
        result.append({"id": f"context-199k-retrieval-{slot}", "kind": "retrieval",
                       "context_sha256": base_sha, "expected": path,
                       "prompt": base + "\n\nTASK: In which supplied FILE does this exact line occur? "
                       + json.dumps(phrase) + " Return only its path, with no explanation.\n"})
    snippet, issue, line = REVIEW[2]
    result.append({"id": "context-199k-review", "kind": "review",
                   "context_sha256": base_sha,
                   "expected": {"issue": issue, "line": line},
                   "prompt": base + "\n\nTASK: Review this separate Python snippet. "
                   "Return only JSON with keys issue and line. The issue must be one of "
                   "zero_division, mutable_default, off_by_one, missing_device. "
                   "Line is the one-based line number causing the bug.\n" + snippet})
    specification, expected = STRUCTURED[2]
    result.append({"id": "context-199k-structured", "kind": "structured",
                   "context_sha256": base_sha, "expected": expected,
                   "prompt": base + "\n\nTASK: " + specification +
                   " Return only a JSON object, with no explanation.\n"})
    if len(result) != 6 or len({row["id"] for row in result}) != 6:
        raise AssertionError("the 199K screen must contain six distinct tasks")
    return result

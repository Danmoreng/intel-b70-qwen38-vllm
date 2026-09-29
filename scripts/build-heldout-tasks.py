#!/usr/bin/env python3
"""Build the frozen C2 task list from the twelve source contexts."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[1]
CONTEXT_ROOT = REPO / "benchmark-results/production-release-v1/heldout-contexts-v1"
OUT = REPO / "benchmarks/heldout-v1/tasks.json"

# Each code task needs both an algorithm and the path of a source line found in
# its own context. The hidden cases score the function and that source lookup.
CODE = [
    ("Define solve(text): return [character, longest_consecutive_run_length]. "
     "Ties use the earliest run; empty text returns ['', 0].",
     [(["aaabbc"], ["a", 3]), (["abbbcc"], ["b", 3]), ([""], ["", 0])]),
    ("Define solve(items): remove duplicate strings while preserving the order "
     "of each item's last appearance.",
     [([["a", "b", "a", "c", "b"]], ["a", "c", "b"]),
      ([[]], []), ([["x", "x", "y"]], ["x", "y"])]),
    ("Define solve(spans): return the number of integer positions covered by "
     "the union of half-open spans [start,end]. Empty or reversed spans cover nothing.",
     [([[[1, 4], [3, 6], [9, 10]]], 6), ([[]], 0),
      ([[[5, 5], [7, 4], [-2, 1]]], 3)]),
    ("Define solve(capacities, budget): repeatedly visit capacities from left "
     "to right, assigning one unit to each unfinished item per round until the "
     "budget is exhausted. Return assigned counts; nonpositive budget gives zeros.",
     [([[2, 1, 3], 4], [2, 1, 1]), ([[0, 2], 3], [0, 2]),
      ([[1, 1], 0], [0, 0])]),
    ("Define solve(lines): parse lines of form key=value with nonempty trimmed "
     "key. Trim surrounding whitespace, ignore lines without '=', and let the "
     "last occurrence win. Return a dict.",
     [([[" a = 1 ", "bad", "a=2", "b = yes"]], {"a": "2", "b": "yes"}),
      ([[]], {}), ([["=x", "k=", "k=v"]], {"k": "v"})]),
    ("Define solve(values, width): return sums of each consecutive window of "
     "positive width. Return [] if width exceeds input or is nonpositive.",
     [([[2, -1, 3, 4], 2], [1, 2, 7]), ([[], 1], []),
      ([[5, 6], 0], []), ([[1, 2, 3], 3], [6])]),
    ("Define solve(value): flatten nested Python lists depth-first into one "
     "list; non-list elements remain unchanged.",
     [([[[1, [2, 3]], 4]], [1, 2, 3, 4]), ([[]], []),
      ([["a", ["b", ["c"]]]], ["a", "b", "c"])]),
    ("Define solve(commands): process ['push', value] and ['pop'] commands on "
     "a stack. Ignore pops on an empty stack. Return the final stack, bottom first.",
     [([[["push", 1], ["push", 2], ["pop"], ["push", 3]]], [1, 3]),
      ([[]], []), ([[["pop"], ["push", "x"]]], ["x"])]),
    ("Define solve(ids, prefix): keep only strings starting with prefix, then "
     "deduplicate in first-appearance order.",
     [([["ab", "x", "abc", "ab"], "ab"], ["ab", "abc"]),
      ([[], "x"], []), ([["a", "b", "a"], ""], ["a", "b"])]),
    ("Define solve(edges, start): edges are directed [from,to] string pairs. "
     "Return all reachable node IDs including start in lexicographic order.",
     [([[["a", "b"], ["b", "c"], ["z", "x"]], "a"], ["a", "b", "c"]),
      ([[], "q"], ["q"]), ([[["x", "x"], ["x", "a"]], "x"], ["a", "x"])]),
    ("Define solve(values, threshold): return [count, sum] for integer values "
     "strictly greater than threshold.",
     [([[1, 3, 3, -2], 2], [2, 6]), ([[], 0], [0, 0]),
      ([[-1, 0, 1], 0], [1, 1])]),
    ("Define solve(values, width): return the number of distinct values in each "
     "consecutive window of positive width. Return [] when width is invalid.",
     [([[1, 2, 1, 3], 2], [2, 2, 2]), ([[], 1], []),
      ([[2, 2, 2], 3], [1]), ([[1, 2], 0], [])]),
]

REVIEW = [
    ("def add(item, seen=[]):\n    seen.append(item)\n    return seen\n", "mutable_default", 1),
    ("def page_index(token, size):  # zero-based token and page\n    return token // size + 1\n", "off_by_one", 2),
    ("def ratio(done, total):  # total may be zero\n    return done / total\n", "zero_division", 2),
    ("def graph_key(device, query_rows, kv_rows):\n    return (query_rows, kv_rows)\n", "missing_device", 2),
    ("def copy_payload(payload):  # nested values must be detached\n    return dict(payload)\n", "shallow_copy", 2),
    ("def read_page(table, page):  # page is a zero-based logical index\n    return table[page - 1]\n", "off_by_one", 2),
    ("def pick(jobs):  # highest numeric priority must come first\n    return sorted(jobs, key=lambda job: job.priority)[0]\n", "wrong_order", 2),
    ("def claim(job):  # attempts counts claims, not completions\n    job.attempts += 1\n    job.attempts += 1\n    return job\n", "double_count", 3),
    ("def accepted_fraction(accepted, proposed):  # proposed can be zero\n    return accepted / proposed\n", "zero_division", 2),
    ("def graph_key(device, image_id, query_rows):  # include device identity\n    return (image_id, query_rows)\n", "missing_device", 2),
    ("def page_of(token, page_size):  # positive page_size; zero-based page IDs\n    return token // page_size + 1\n", "off_by_one", 2),
    ("def restore_payload(saved):  # restored nested values must be independent\n    return dict(saved)\n", "shallow_copy", 2),
]


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_lines(context: str) -> list[tuple[float, str, str]]:
    blocks = list(re.finditer(r"^### FILE ([^\n]+)\n", context, re.MULTILINE))
    rows = []
    for index, block in enumerate(blocks):
        end = blocks[index + 1].start() if index + 1 < len(blocks) else len(context)
        for line in context[block.end():end].splitlines():
            phrase = line.strip()
            if (40 <= len(phrase) <= 105 and context.count(phrase) == 1
                    and not phrase.startswith(("#", "//", "*"))):
                rows.append((block.start() / len(context), block.group(1), phrase))
                break
    if len(rows) < 2:
        raise RuntimeError("too few unique source lines")
    return rows


def pick(rows: list[tuple[float, str, str]], fraction: float):
    return min(rows, key=lambda row: abs(row[0] - fraction))


def main() -> None:
    context_manifest = json.loads((CONTEXT_ROOT / "manifest.json").read_text())
    agents = json.loads((OUT.parent / "agent_tasks.json").read_text())
    tasks = []
    for agent in agents:
        tasks.append({"id": agent["id"], "kind": "agent_code",
                      "context_id": f"context-{agent['context_index']:02d}",
                      "instruction": agent["instruction"],
                      "acceptance": agent["acceptance"], "critical": True})
    retrieval_contexts = (3, 5, 7, 8, 9, 10, 11, 12)
    tool_contexts = (1, 2, 4, 6, 9, 10, 11, 12)
    for index, context_row in enumerate(context_manifest["contexts"], 1):
        context = (CONTEXT_ROOT / context_row["filename"]).read_text()
        if sha(context.encode()) != context_row["source_context_sha256"]:
            raise RuntimeError(f"context hash mismatch: {index}")
        lines = source_lines(context)
        code_path, code_phrase = pick(lines, 0.25 + (index % 3) * 0.2)[1:]
        instruction, cases = CODE[index - 1]
        tasks.append({"id": f"code-{index:02d}", "kind": "code",
                      "context_id": context_row["id"], "critical": index >= 11,
                      "instruction": instruction + " Instead of returning that value directly, return a dict with exactly keys "
                      "source_file and result: result is the specified value; "
                      "source_file is the FILE path in the supplied source "
                      "containing this exact line: " + json.dumps(code_phrase, ensure_ascii=False)
                      + ". Return only Python code defining solve.\n",
                      "expected_source_file": code_path, "cases": cases})
        review_path, review_phrase = pick(lines, 0.75)[1:]
        snippet, issue, line = REVIEW[index - 1]
        tasks.append({"id": f"review-{index:02d}", "kind": "review",
                      "context_id": context_row["id"], "critical": index >= 11,
                      "instruction": "Review the separate proposed Python snippet below. "
                      "Return only a JSON object with keys issue, line, source_file. "
                      "issue must be one of mutable_default, off_by_one, "
                      "zero_division, missing_device, shallow_copy, wrong_order, "
                      "double_count; line is one-based. source_file is the FILE "
                      "path in the supplied source containing this exact line: "
                      + json.dumps(review_phrase, ensure_ascii=False) + "\n" + snippet,
                      "expected": {"issue": issue, "line": line,
                                   "source_file": review_path}})
        if index in retrieval_contexts:
            path, phrase = pick(lines, (0.05, 0.50, 0.95)[index % 3])[1:]
            tasks.append({"id": f"retrieval-{index:02d}", "kind": "retrieval",
                          "context_id": context_row["id"], "critical": index >= 11,
                          "instruction": "Which supplied FILE contains this exact line? "
                          + json.dumps(phrase, ensure_ascii=False)
                          + " Return only its path.\n", "expected": path})
        if index in tool_contexts:
            path, phrase = pick(lines, (0.15, 0.55, 0.85)[index % 3])[1:]
            tasks.append({"id": f"tool-{index:02d}", "kind": "tool",
                          "context_id": context_row["id"], "critical": index >= 11,
                          "instruction": "Find the supplied FILE containing this exact line: "
                          + json.dumps(phrase, ensure_ascii=False)
                          + ". Call identify_source exactly once with its path "
                          "and the exact line as marker. Do not invent a path.\n",
                          "expected": {"path": path, "marker": phrase}})
    kinds = Counter(task["kind"] for task in tasks)
    if kinds != {"agent_code": 8, "code": 12, "review": 12,
                 "retrieval": 8, "tool": 8} or len(tasks) != 48:
        raise RuntimeError(f"invalid held-out task distribution: {kinds}")
    payload = {"schema": 1, "context_manifest_sha256": sha(
        (CONTEXT_ROOT / "manifest.json").read_bytes()), "tasks": tasks}
    if OUT.exists():
        raise RuntimeError("frozen held-out tasks already exist")
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"tasks": len(tasks), "categories": kinds,
                      "sha256": sha(OUT.read_bytes())}))


if __name__ == "__main__":
    main()

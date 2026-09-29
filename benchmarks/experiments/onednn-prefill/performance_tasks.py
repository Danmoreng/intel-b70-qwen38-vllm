"""Frozen, automatically scored task screen for the two attention profiles.

Each of five real source-review contexts is followed by two coding tasks, two
source-retrieval tasks, one code review, and one structured-output task. The
same context is deliberately reused within a group to exercise prefix caching.
"""

import hashlib
import json
import os
import re
from pathlib import Path


FROZEN = Path(os.environ["B70_FROZEN_FIXTURE_ROOT"]).resolve()

CODE = [
    ("Define solve(events). Each event is [request_id, emitted_tokens]. Sum tokens "
     "by request and return [request_id, total] pairs in order of first appearance.",
     [([[["a", 2], ["b", 3], ["a", 4]]], [["a", 6], ["b", 3]]),
      ([[]], []), ([ [["x", 0], ["x", 5], ["z", -1]] ], [["x", 5], ["z", -1]])]),
    ("Define solve(intervals). Merge overlapping or touching half-open integer "
     "intervals [start, end], and return sorted merged intervals. An empty input returns [].",
     [([[[1, 4], [4, 7], [10, 11], [2, 5]]], [[1, 7], [10, 11]]),
      ([[]], []), ([ [[-3, -1], [0, 2], [2, 2]] ], [[-3, -1], [0, 2]])]),
    ("Define solve(keys, capacity). Simulate an LRU cache: each key access is a hit "
     "if present, then becomes most recent. Evict the least recent key when needed. "
     "Return {'hits': integer, 'cache': keys from least to most recent}. Capacity 0 is valid.",
     [([["a", "b", "a", "c", "b"], 2], {"hits": 1, "cache": ["c", "b"]}),
      ([[], 3], {"hits": 0, "cache": []}),
      ([["x", "x"], 0], {"hits": 0, "cache": []})]),
    ("Define solve(lines, metric). Parse Prometheus sample lines of the form "
     "metric_name{optional labels} number or metric_name number. Ignore blank and "
     "comment lines. Sum numeric values whose metric name exactly matches metric. "
     "Return a float. Labels must not change the name match.",
     [([["# HELP x", "x{a=\"1\"} 2.5", "xy 100", "x 1.5"], "x"], 4.0),
      ([[], "x"], 0.0),
      ([["v 1e2", "v{a=\"2\"} -2.5", "other 4"], "v"], 97.5)]),
    ("Define solve(values, width). Return the maximum of every consecutive "
     "window of width elements, in left-to-right order. Return [] when width "
     "is nonpositive or larger than the input.",
     [([[3, 1, 5, 2, 4], 3], [5, 5, 5]),
      ([[], 1], []), ([[-2, -1, -3], 1], [-2, -1, -3]),
      ([[1, 2], 0], [])]),
    ("Define solve(graph, start). graph maps each node string to a list of "
     "outgoing neighbor strings. Return a dict of shortest directed edge counts "
     "from start to every reachable node, including start at 0. A neighbor need "
     "not itself be a key in graph.",
     [([{"a": ["b", "c"], "b": ["d"], "c": ["d"]}, "a"],
       {"a": 0, "b": 1, "c": 1, "d": 2}),
      ([{}, "z"], {"z": 0}),
      ([{"a": ["a", "b"], "b": []}, "a"], {"a": 0, "b": 1})]),
    ("Define solve(lengths, budget). Allocate at most budget nonnegative tokens "
     "to requests in input order, at most each requested length. Return one "
     "allocated count per request. A nonpositive budget allocates zero everywhere.",
     [([[5, 4, 8], 10], [5, 4, 1]),
      ([[3, 2], 0], [0, 0]),
      ([[], 9], []), ([[2, 2, 2], 3], [2, 1, 0])]),
    ("Define solve(text). Check (), [], and {} delimiters while ignoring other "
     "characters. Return -1 if balanced. Return the zero-based index of the "
     "first unmatched closing delimiter or mismatched closing delimiter. "
     "If only openings remain, return len(text).",
     [(["a(b[c]{d})"], -1), (["([)]"], 2),
      (["(()"], 3), (["x]"], 1), ([""], -1)]),
    ("Define solve(items, k). Each item is [identifier_string, numeric_score]. "
     "Return identifiers of the k highest scores, breaking score ties by "
     "identifier in ascending lexicographic order. Return [] for nonpositive k.",
     [([[["b", 4], ["a", 4], ["c", 9]], 2], ["c", "a"]),
      ([[], 3], []), ([ [["x", -1], ["y", 0]] , 0], []),
      ([ [["b", 1], ["a", 2]] , 5], ["a", "b"])]),
    ("Define solve(edges). edges is a list of [before, after] string pairs. "
     "Return the lexicographically smallest topological order of all nodes "
     "mentioned. If the graph has a cycle, return [].",
     [([[["a", "c"], ["b", "c"]]], ["a", "b", "c"]),
      ([[]], []), ([[["x", "y"], ["y", "x"]]], []),
      ([[["z", "a"], ["a", "b"]]], ["z", "a", "b"])])
]

REVIEW = [
    ("def rate(done, total):\n    return done / total\n", "zero_division", 2),
    ("def add(item, seen=[]):\n    seen.append(item)\n    return seen\n", "mutable_default", 1),
    ("def page_of(token, page_size):  # positive page_size; zero-based page IDs\n"
     "    return token // page_size + 1\n", "off_by_one", 2),
    ("def graph_key(device, q, k):\n    return (q, k)\n", "missing_device", 2),
    ("def readable(table, logical_page):  # logical_page is zero-based\n"
     "    return table[logical_page - 1]\n", "off_by_one", 2),
]

STRUCTURED = [
    ("Given 672 accepted draft tokens and 1404 proposed draft tokens, return "
     "JSON with exactly keys accepted and fraction; fraction is accepted/proposed "
     "rounded to four decimal places.", {"accepted": 672, "fraction": 0.4786}),
    ("Given 32722 prompt tokens and 29952 cached tokens, return JSON with "
     "exactly keys computed and cached; computed is prompt minus cached.",
     {"computed": 2770, "cached": 29952}),
    ("Given 32768 tokens in 20 seconds, return JSON with exactly keys "
     "tokens_per_second and seconds_per_1024_tokens, both numeric.",
     {"tokens_per_second": 1638.4, "seconds_per_1024_tokens": 0.625}),
    ("Given prefill 20.5 seconds, decode 10.6 seconds, and overhead 0.4 "
     "seconds, return JSON with exactly keys total_seconds and decode_fraction, "
     "where decode_fraction is decode/total rounded to four decimals.",
     {"total_seconds": 31.5, "decode_fraction": 0.3365}),
    ("A page holds 1664 tokens. A prompt has 4993 tokens. Return JSON with "
     "exactly keys pages_needed and tokens_in_last_page.",
     {"pages_needed": 4, "tokens_in_last_page": 1}),
]


def contexts():
    result = []
    for repeat in range(1, 6):
        prompt = (FROZEN / f"phase-32k-c1-r{repeat}" / "prompt-1.txt").read_text()
        # The benchmark's task is the final paragraph after its source blocks.
        base, ending = prompt.rsplit("\n\n", 1)
        if not any(ending.startswith(prefix) for prefix in
                   ("Review the supplied", "Explain the purpose",
                    "Write a focused", "Write a detailed")):
            raise ValueError(f"unexpected frozen prompt ending in context {repeat}")
        result.append(base)
    return result


def retrievals(base):
    blocks = re.findall(r"### FILE ([^\n]+)\n(.*?)(?=\n### FILE |\Z)",
                        base, flags=re.DOTALL)
    if len(blocks) < 4:
        raise ValueError("source context contains too few files")
    choices = []
    for target in (len(blocks) // 3, 2 * len(blocks) // 3):
        for distance in range(len(blocks)):
            path, content = blocks[(target + distance) % len(blocks)]
            for line in content.splitlines():
                phrase = line.strip()
                if (35 <= len(phrase) <= 90 and phrase.count('"') < 2
                        and base.count(phrase) == 1 and phrase not in choices):
                    choices.append((path, phrase))
                    break
            if len(choices) == (1 if target == len(blocks) // 3 else 2):
                break
    if len(choices) != 2:
        raise ValueError("could not find two unique source lines")
    return choices


def tasks():
    result = []
    for context_index, base in enumerate(contexts()):
        base_sha = hashlib.sha256(base.encode()).hexdigest()
        prefix = f"context-{context_index + 1}"
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
    if len(result) != 30 or len({row["id"] for row in result}) != 30:
        raise AssertionError("the practical screen must contain 30 distinct tasks")
    return result

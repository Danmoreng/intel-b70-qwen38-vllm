#!/usr/bin/env python3
"""Small serving regression suite, not a general quantization quality benchmark."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request


TASKS = [
    ("merge_intervals", "Implement merge_intervals(intervals): merge overlapping OR touching closed integer intervals, sort by start, return a list of tuples. Do not mutate the input. Empty input returns [].",
     "assert merge_intervals([])==[]\na=[(5,8),(1,3),(3,6),(12,12)]; b=a.copy(); assert merge_intervals(a)==[(1,8),(12,12)]; assert a==b\nassert merge_intervals([(2,2),(2,2),(-5,-1)])==[(-5,-1),(2,2)]\nassert merge_intervals([(1,10),(2,3),(11,12)])==[(1,10),(11,12)]"),
    ("stable_unique", "Implement stable_unique(items): return a list retaining only the first occurrence of each hashable item, in input order. Accept any iterable, do not mutate inputs, use O(n) expected time.",
     "assert stable_unique([])==[]\na=[3,1,3,2,1]; assert stable_unique(a)==[3,1,2]; assert a==[3,1,3,2,1]\nassert stable_unique(iter(['a','b','a']))==['a','b']\nassert stable_unique([None,None,(1,2),(1,2)])==[None,(1,2)]"),
    ("parse_pairs", "Implement parse_pairs(text): parse nonempty lines of key=value, stripping whitespace around keys/values, splitting only at the FIRST equals sign. Ignore blank lines and lines whose stripped form starts with #. Raise ValueError on a non-comment line missing =, an empty key, or a duplicate key. Return dict[str,str]. Empty values are valid.",
     "assert parse_pairs('')=={}\nassert parse_pairs(' # comment\\n a = 1 \\nb=x=y\\nc=\\n')=={'a':'1','b':'x=y','c':''}\nfor bad in ['a', '=v', 'a=1\\na=2']:\n try: parse_pairs(bad)\n except ValueError: pass\n else: raise AssertionError(bad)"),
]
NLL_TEXTS = [
    "Berlin is the capital of Germany. Paris is the capital of France. A week contains seven days. Water freezes at zero degrees Celsius under standard atmospheric pressure.",
    "def stable_unique(items):\n    seen = set()\n    result = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result\n",
    "Bei einem Cache entscheidet die Gültigkeitsdauer darüber, ob ein Eintrag noch verwendet werden darf. Ein abgelaufener Eintrag muss beim Lesen entfernt werden. Tests sollten auch den exakten Zeitpunkt des Ablaufs prüfen.",
]


def post(root, endpoint, payload):
    request = urllib.request.Request(root + endpoint, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="http://127.0.0.1:18087")
    parser.add_argument("--image", required=True, help="CPU-only sandbox image for generated code")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"scope": "three small coding tasks and three short teacher-forced NLL samples", "coding": [], "nll": []}

    def save():
        (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    for name, task, assertions in TASKS:
        payload = {"model": "Qwen3.8-27B", "messages": [{"role": "user", "content": task + " Return only Python code, no prose."}],
                   "temperature": 0, "max_tokens": 1536, "chat_template_kwargs": {"enable_thinking": False}}
        start = time.monotonic()
        response = post(args.root, "/v1/chat/completions", payload)
        (args.output_dir / f"{name}-response.json").write_text(json.dumps(response, indent=2))
        choice = response["choices"][0]
        answer = choice["message"].get("content") or ""
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", answer, re.S)
        code = "\n".join(blocks) if blocks else answer
        program = code + "\n" + assertions + "\nprint('ALL_ASSERTIONS_PASSED')\n"
        (args.output_dir / f"{name}.py").write_text(program)
        checked = subprocess.run([
            "docker", "run", "--rm", "-i", "--network=none", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--memory=512m",
            "--pids-limit=64", "--cpus=1", "--user=65534:65534", "--entrypoint=python",
            args.image, "-I", "-"], input=program, text=True, capture_output=True, timeout=30)
        result["coding"].append({"name": name, "pass": checked.returncode == 0 and choice["finish_reason"] != "length",
                                 "finish_reason": choice["finish_reason"], "usage": response.get("usage"),
                                 "wall_s": time.monotonic()-start, "stdout": checked.stdout, "stderr": checked.stderr})
        save()
    for index, text in enumerate(NLL_TEXTS):
        payload = {"model": "Qwen3.8-27B", "prompt": text, "max_tokens": 1,
                   "temperature": 0, "prompt_logprobs": 1, "return_token_ids": True}
        try:
            response = post(args.root, "/v1/completions", payload)
            (args.output_dir / f"nll-{index}.json").write_text(json.dumps(response, indent=2))
            choice = response["choices"][0]
            entries = choice.get("prompt_logprobs") or response.get("prompt_logprobs")
            token_ids = choice.get("prompt_token_ids") or response.get("prompt_token_ids")
            if not entries or not token_ids:
                result["nll"].append({"index": index, "status": "unavailable", "reason": "API omitted prompt token IDs/logprobs"})
            else:
                values = [entry[str(token)]["logprob"] for token, entry in zip(token_ids, entries) if entry is not None]
                result["nll"].append({"index": index, "status": "measured", "tokens": len(values), "nll": -sum(values)/len(values)})
        except urllib.error.HTTPError as error:
            result["nll"].append({"index": index, "status": "unavailable", "reason": error.read().decode()})
        save()
    result["coding_pass"] = all(row["pass"] for row in result["coding"])
    save()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

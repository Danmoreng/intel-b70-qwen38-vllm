#!/usr/bin/env python3
"""Pair a continuing 4K decoder with a newly arriving cold 32K prefill."""

import argparse
import hashlib
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path

from replay_frozen import FROZEN, METRICS, snapshot


def payload(case):
    folder = FROZEN / f"phase-{case}-c1-r1"
    frozen = json.loads((folder / "request-1.json").read_text())
    prompt = (folder / "prompt-1.txt").read_text()
    if hashlib.sha256(prompt.encode()).hexdigest() != frozen["prompt_sha256"]:
        raise ValueError(f"{case} frozen prompt hash changed")
    result = {key: frozen[key] for key in (
        "model", "max_tokens", "temperature", "top_p", "top_k", "seed",
        "ignore_eos", "chat_template_kwargs", "stream", "stream_options")}
    result["messages"] = [{"role": "user", "content": prompt}]
    return result, frozen["prompt_sha256"]


def stream(base, body, result, started):
    request = urllib.request.Request(base + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    result["start_s"] = time.monotonic() - started
    pieces, stamps = [], []
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            for line in response:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("error"):
                    raise RuntimeError(event["error"])
                result["usage"] = event.get("usage") or result.get("usage")
                for choice in event.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if piece:
                        pieces.append(piece)
                        stamps.append(time.monotonic() - started)
                        result["first_piece"].set()
                    result["finish_reason"] = (choice.get("finish_reason") or
                                               result.get("finish_reason"))
        result["end_s"] = time.monotonic() - started
        result["output_sha256"] = hashlib.sha256("".join(pieces).encode()).hexdigest()
        result["stream_piece_times_s"] = stamps
        result["stream_piece_count"] = len(stamps)
    except Exception as error:
        result["error"] = repr(error)
        result["first_piece"].set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    short_body, short_hash = payload("4k")
    long_body, long_hash = payload("32k")
    before = snapshot(args.base)
    started = time.monotonic()
    short = {"first_piece": threading.Event()}
    long = {"first_piece": threading.Event()}
    short_thread = threading.Thread(target=stream,
        args=(args.base, short_body, short, started), daemon=True)
    long_thread = threading.Thread(target=stream,
        args=(args.base, long_body, long, started), daemon=True)
    short_thread.start()
    if not short["first_piece"].wait(timeout=120) or "error" in short:
        raise RuntimeError(f"short request did not start decoding: {short.get('error')}")
    time.sleep(0.1)
    long_thread.start()
    short_thread.join(timeout=900)
    long_thread.join(timeout=900)
    if short_thread.is_alive() or long_thread.is_alive():
        raise TimeoutError("staggered request did not finish")
    if "error" in short or "error" in long:
        raise RuntimeError(f"staggered request failed: {short.get('error')} / {long.get('error')}")
    after = snapshot(args.base)
    metrics = {key: after[key] - before[key] for key in METRICS}
    long_start, long_first = long["start_s"], long["stream_piece_times_s"][0]
    stamps = short["stream_piece_times_s"]
    gaps = [(right - left, left, right) for left, right in zip(stamps, stamps[1:])]
    during = [gap for gap in gaps if gap[1] <= long_first and gap[2] >= long_start]
    result = {
        "arm": args.arm, "short_prompt_sha256": short_hash,
        "long_prompt_sha256": long_hash,
        "metrics": metrics,
        "short": {key: value for key, value in short.items() if key != "first_piece"},
        "long": {key: value for key, value in long.items() if key != "first_piece"},
        "short_max_stream_gap_s": max((gap[0] for gap in gaps), default=None),
        "short_median_stream_gap_s": statistics.median(gap[0] for gap in gaps) if gaps else None,
        "short_max_gap_overlapping_long_prefill_s": max(
            (gap[0] for gap in during), default=None),
        "long_ttft_s": long_first - long_start,
        "short_wall_s": short["end_s"] - short["start_s"],
        "long_wall_s": long["end_s"] - long_start,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("short", "long")}), flush=True)
    if metrics["preemptions"] or metrics["cached_tokens"]:
        raise RuntimeError("staggered cold pair had cache hit or preemption")
    for request in (short, long):
        if request["usage"]["completion_tokens"] != 1024:
            raise RuntimeError("frozen request did not emit 1024 tokens")


if __name__ == "__main__":
    main()

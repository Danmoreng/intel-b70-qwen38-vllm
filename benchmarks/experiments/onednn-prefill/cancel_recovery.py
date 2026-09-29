#!/usr/bin/env python3
"""Cancel an in-flight 128K prefill, then verify short and oneDNN recovery."""

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from replay_frozen import FROZEN, METRICS, snapshot


def frozen_payload(case, repeat):
    folder = FROZEN / f"phase-{case}-c1-r{repeat}"
    frozen = json.loads((folder / "request-1.json").read_text())
    prompt = (folder / "prompt-1.txt").read_text()
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    if digest != frozen["prompt_sha256"]:
        raise ValueError(f"frozen prompt hash changed: {folder}")
    payload = {key: frozen[key] for key in (
        "model", "max_tokens", "temperature", "top_p", "top_k", "seed",
        "ignore_eos", "chat_template_kwargs", "stream", "stream_options")}
    payload["messages"] = [{"role": "user", "content": prompt}]
    return payload, digest


def recovery_request(base, payload):
    payload = {**payload, "max_tokens": 128, "stream": False}
    payload.pop("stream_options", None)
    request = urllib.request.Request(base + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.load(response)
    text = data["choices"][0]["message"].get("content") or ""
    return {
        "wall_s": time.monotonic() - started,
        "usage": data["usage"],
        "finish_reason": data["choices"][0]["finish_reason"],
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--cancel-after", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cancel_after <= 0:
        parser.error("cancel-after must be positive")
    long_payload, long_hash = frozen_payload("128k", 3)
    short_payload, short_hash = frozen_payload("4k", 1)
    check_payload, check_hash = frozen_payload("32k", 1)
    before = snapshot(args.base)
    with tempfile.TemporaryDirectory(prefix="b70-cancel-") as temporary:
        path = Path(temporary) / "request.json"
        path.write_text(json.dumps(long_payload))
        process = subprocess.Popen([
            "curl", "--silent", "--show-error", "--no-buffer", "--max-time", "900",
            "-H", "Content-Type: application/json", "--data-binary", f"@{path}",
            args.base + "/v1/chat/completions"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            time.sleep(args.cancel_after)
            if process.poll() is not None:
                raise RuntimeError("long request ended before cancellation")
            process.terminate()
            _, error = process.communicate(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
    time.sleep(2)
    with urllib.request.urlopen(args.base + "/health", timeout=10):
        pass
    after_cancel = snapshot(args.base)
    short = recovery_request(args.base, short_payload)
    long = recovery_request(args.base, check_payload)
    after_recovery = snapshot(args.base)
    result = {
        "arm": args.arm, "cancel_after_s": args.cancel_after,
        "cancelled_prompt_sha256": long_hash,
        "cancelled_process_returncode": process.returncode,
        "cancelled_process_stderr": error.decode(errors="replace")[-500:],
        "short_prompt_sha256": short_hash,
        "long_recovery_prompt_sha256": check_hash,
        "metrics_after_cancel": {key: after_cancel[key] - before[key] for key in METRICS},
        "metrics_recovery": {key: after_recovery[key] - after_cancel[key] for key in METRICS},
        "short_recovery": short, "long_recovery": long,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if any(request["usage"]["completion_tokens"] != 128 for request in (short, long)):
        raise RuntimeError("recovery request did not emit 128 tokens")
    if result["metrics_recovery"]["preemptions"]:
        raise RuntimeError("recovery preempted")


if __name__ == "__main__":
    main()

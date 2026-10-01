#!/usr/bin/env python3
"""Record a cold long-context EXL3 request, reporting preemptions explicitly."""

import argparse
import fcntl
import importlib.util
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "campaign", REPO / "scripts/run-web-coding-campaign.py"
)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
R = C.R


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request", type=int, default=94)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("fresh diagnostic output required")
    original = json.loads(
        (args.source / f"original-request-{args.request:04d}.json").read_text()
    )
    with (REPO.parent / "Local-AI-B70/qwen38/context-benchmark/run.lock").open(
        "a"
    ) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True)
        (args.output / "diagnostic-source.py").write_bytes(Path(__file__).read_bytes())
        result = {
            "status": "running",
            "purpose": "Diagnostic cold throughput: scheduler preemptions are reported, not accepted as a clean zero-preemption control",
            "started_at": R.now(),
            "source_request": args.request,
        }
        R.save(args.output / "diagnostic.json", result)
        try:
            C.command(["systemctl", "--user", "stop", C.SERVICE])
            launch = C.start_exl()
            identity = C.wait_health("http://127.0.0.1:8082", C.EXL_NAME, C.EXL_IMAGE)
            model = R.http("http://127.0.0.1:8082", "/v1/models")["data"][0]["id"]
            warm = args.output / "warmup"
            warm.mkdir()
            C.warmup("http://127.0.0.1:8082", model, warm)
            R.cap()
            payload = dict(original)
            payload.update(
                model=model,
                max_tokens=1024,
                ignore_eos=True,
                cache_salt="flappy-v4-common-prefix-20261001-" + str(args.request),
            )
            preflight = R.bound_output("http://127.0.0.1:8082", payload)
            before = R.snapshot("http://127.0.0.1:8082")
            started = time.monotonic()
            response = R.stream(
                "http://127.0.0.1:8082", payload, args.output / "response.sse.jsonl"
            )
            after, native = R.wait_accounted(
                "http://127.0.0.1:8082", before, response["usage"]
            )
            R.save(args.output / "request.json", payload)
            R.save(args.output / "response.json", response)
            result.update(
                identity=identity,
                launch=launch,
                usage=response["usage"],
                native=native,
                native_before=before,
                native_after=after,
                tokenize_preflight=preflight,
                wall_s=time.monotonic() - started,
                prefill_tps=native["prefill_tokens"] / native["prefill_seconds"],
                decode_tps=1023 / native["decode_seconds"],
            )
            R.save(args.output / "diagnostic.json", result)
            if (
                native["cached_tokens"] != 0
                or native["prefill_tokens"] != response["usage"]["prompt_tokens"]
                or preflight["prompt_tokens"] != response["usage"]["prompt_tokens"]
                or response["usage"]["completion_tokens"] != 1024
                or R.identity(C.EXL_NAME) != identity
            ):
                raise RuntimeError("diagnostic token accounting or identity differs")
            result.update(status="complete", finished_at=R.now())
        except BaseException as error:
            result.update(
                status="failed",
                error=type(error).__name__ + ": " + str(error),
                finished_at=R.now(),
            )
            raise
        finally:
            R.save(args.output / "diagnostic.json", result)
            R.save(args.output / "production-restored.json", C.restore())


if __name__ == "__main__":
    main()

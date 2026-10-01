#!/usr/bin/env python3
"""Three identical coding histories, cold KV and fixed-length throughput only."""

import argparse
import fcntl
import importlib.util
import json
import shutil
import signal
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
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Preserve validated measurements and retry unfinished EXL3 points on a fresh worker",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("fresh replay output required")
    rows = [
        json.loads(x) for x in (args.source / "requests.jsonl").read_text().splitlines()
    ]
    selected = [
        min(rows, key=lambda r: abs(r["usage"]["prompt_tokens"] - target))
        for target in [100000, 140000, 190000]
    ]
    originals = []
    for row in selected:
        file = args.source / f"request-{row['request']:04d}.json"
        if R.sha(file) != row["request_sha256"]:
            raise RuntimeError("source request hash differs")
        originals.append(json.loads(file.read_text()))
    lock_path = REPO.parent / "Local-AI-B70/qwen38/context-benchmark/run.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True)
        (args.output / "replay-source.py").write_bytes(Path(__file__).read_bytes())
        for row, payload in zip(selected, originals, strict=True):
            R.save(args.output / f"original-request-{row['request']:04d}.json", payload)
        release = json.loads((REPO / "config/production_image.json").read_text())
        state = {
            "status": "running",
            "started_at": R.now(),
            "purpose": "Fixed-length throughput control using identical recorded coding histories; no new coding-agent trajectory or quality score",
            "source_fixture_manifest_sha256": json.loads(
                (args.source / "provenance.json").read_text()
            )["fixture_manifest_sha256"],
            "selected_requests": [
                {
                    "request": r["request"],
                    "original_context": r["usage"]["prompt_tokens"],
                    "sha256": r["request_sha256"],
                }
                for r in selected
            ],
            "engines": {},
            "output_tokens": 1024,
            "ignore_eos": True,
            "cache": "A different explicit cache_salt for each source prompt; native cached tokens must equal zero",
            "power_w": 180,
        }
        if args.resume_from:
            previous = json.loads((args.resume_from / "replay.json").read_text())
            if (
                previous["selected_requests"] != state["selected_requests"]
                or previous["status"] != "failed"
            ):
                raise RuntimeError("resume source must be this failed control")
            state["engines"] = previous["engines"]
            state["resumed_from"] = str(args.resume_from.resolve())
            state["previous_error"] = previous["error"]
            for name in previous["engines"]:
                shutil.copytree(args.resume_from / name, args.output / name)
        R.save(args.output / "replay.json", state)

        def interrupted(*_):
            raise RuntimeError("replay interrupted")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            for engine in ["gptq", "exl3"]:
                if len(state["engines"].get(engine, {}).get("requests", [])) == len(
                    selected
                ):
                    continue
                state["phase"] = engine + "-startup"
                R.save(args.output / "replay.json", state)
                if engine == "gptq":
                    C.stop_exl()
                    C.command(["systemctl", "--user", "restart", C.SERVICE])
                    base, container, image = (
                        "http://127.0.0.1:8081",
                        "b70-qwen38-vllm",
                        release["image_id"],
                    )
                else:
                    C.command(["systemctl", "--user", "stop", C.SERVICE])
                    launch = C.start_exl()
                    R.save(args.output / "exl3-launch.json", {"argv": launch})
                    base, container, image = (
                        "http://127.0.0.1:8082",
                        C.EXL_NAME,
                        C.EXL_IMAGE,
                    )
                identity = C.wait_health(base, container, image)
                # Verify support rather than silently accepting an ignored field.
                C.command(
                    [
                        "docker",
                        "exec",
                        container,
                        "python",
                        "-c",
                        'from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest; assert "cache_salt" in ChatCompletionRequest.model_fields',
                    ]
                )
                model = R.http(base, "/v1/models")["data"][0]["id"]
                root = args.output / engine
                root.mkdir(exist_ok=True)
                warm = root / ("warmup-continuation" if args.resume_from else "warmup")
                warm.mkdir()
                C.warmup(base, model, warm)
                results = state["engines"].get(engine, {}).get("requests", [])
                if engine not in state["engines"]:
                    state["engines"][engine] = {
                        "identity": identity,
                        "requests": results,
                    }
                else:
                    state["engines"][engine].setdefault(
                        "continuation_workers", []
                    ).append({"identity": identity, "excluded_warmup": str(warm)})
                for row, original in zip(selected, originals, strict=True):
                    if any(x["source_request"] == row["request"] for x in results):
                        continue
                    R.cap()
                    if R.identity(container) != identity:
                        raise RuntimeError("worker changed")
                    payload = dict(original)
                    payload.update(
                        model=model,
                        max_tokens=1024,
                        ignore_eos=True,
                        cache_salt="flappy-v4-common-prefix-20261001-"
                        + str(row["request"]),
                    )
                    preflight = R.bound_output(base, payload)
                    if payload["max_tokens"] != 1024:
                        raise RuntimeError("not enough shared output capacity")
                    before = R.snapshot(base)
                    started = time.monotonic()
                    response = R.stream(
                        base, payload, root / f"request-{row['request']:04d}.sse.jsonl"
                    )
                    after, native = R.wait_accounted(base, before, response["usage"])
                    R.save(root / f"request-{row['request']:04d}.json", payload)
                    R.save(root / f"response-{row['request']:04d}.json", response)
                    R.save(
                        root / f"observation-{row['request']:04d}.json",
                        {
                            "identity": identity,
                            "usage": response["usage"],
                            "native": native,
                            "native_before": before,
                            "native_after": after,
                            "tokenize_preflight": preflight,
                        },
                    )
                    if (
                        response["usage"]["prompt_tokens"] != preflight["prompt_tokens"]
                        or response["usage"]["completion_tokens"] != 1024
                        or native["cached_tokens"] != 0
                        or native["prefill_tokens"]
                        != response["usage"]["prompt_tokens"]
                        or native["preemptions"] != 0
                    ):
                        raise RuntimeError(
                            "replay is not exact-count cold throughput: "
                            + json.dumps(
                                {
                                    "usage": response["usage"],
                                    "native": native,
                                    "preflight": preflight,
                                }
                            )
                        )
                    R.save(root / f"request-{row['request']:04d}.json", payload)
                    R.save(root / f"response-{row['request']:04d}.json", response)
                    results.append(
                        {
                            "identity": identity,
                            "source_request": row["request"],
                            "source_request_sha256": row["request_sha256"],
                            "usage": response["usage"],
                            "native": native,
                            "native_before": before,
                            "native_after": after,
                            "tokenize_preflight": preflight,
                            "prefill_tps": native["prefill_tokens"]
                            / native["prefill_seconds"],
                            "decode_tps": 1023 / native["decode_seconds"],
                            "wall_s": time.monotonic() - started,
                        }
                    )
                    state["phase"] = engine + "-replay"
                    R.save(args.output / "replay.json", state)
            state.update(status="complete", finished_at=R.now())
            R.save(args.output / "replay.json", state)
        except BaseException as e:
            state.update(
                status="failed",
                error=type(e).__name__ + ": " + str(e),
                finished_at=R.now(),
            )
            R.save(args.output / "replay.json", state)
            raise
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            R.save(args.output / "production-restored.json", C.restore())


if __name__ == "__main__":
    main()

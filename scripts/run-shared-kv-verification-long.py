#!/usr/bin/env python3
"""Run a fresh full-context control/M04 comparison and restore production."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time


REPO = Path(__file__).resolve().parents[1]
CONTROL_IMAGE = "sha256:3f20b0cf493fe0904a7efd0ca310067790e901bc5d60203340c411e57c25010a"
CANDIDATE_IMAGE = "local/qwen38-b70-vllm:m04-shared-kv-candidate"
CANDIDATE_ID = "sha256:aee9857bef1f37c8f0ee136d9f89d7166201212175a8b171d958627706cf1c0b"
spec = importlib.util.spec_from_file_location(
    "b70_diagnostics", REPO / "scripts/run-diagnostics.py")
diag = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(diag)


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    state = {"status": "preflight", "started_at_unix": time.time(),
             "run_dir": str(run)}
    save(run / "state.json", state)
    lock_path = Path("/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock")
    lock = lock_path.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,
                  lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    inspect = diag.production_inspect()
    if inspect["Image"] != CONTROL_IMAGE:
        raise RuntimeError("production image changed")
    diag.assert_mtp4(inspect["Config"]["Cmd"])
    actual = subprocess.check_output(
        ["docker", "image", "inspect", CANDIDATE_IMAGE, "--format", "{{.Id}}"],
        text=True).strip()
    if actual != CANDIDATE_ID:
        raise RuntimeError(f"candidate image changed: {actual}")
    if int(diag.POWER_CAP.read_text()) != 180000000:
        raise RuntimeError("power cap is not 180 W")
    diag.ensure_idle(diag.PRODUCTION_URL)
    save(run / "production-inspect.json", inspect)
    try:
        state["status"] = "stopping-production"
        save(run / "state.json", state)
        subprocess.run(["systemctl", "--user", "stop",
                        diag.PRODUCTION_SERVICE], check=True, timeout=120)
        state["status"] = "control-196k"
        save(run / "state.json", state)
        control = diag.run_long_context(
            inspect, run, phase_name="control-196k",
            container_name="b70-m04-control-196k", image=CONTROL_IMAGE,
            payload_overrides={"ignore_eos": True})
        state["status"] = "candidate-196k"
        save(run / "state.json", state)
        candidate = diag.run_long_context(
            inspect, run, phase_name="candidate-196k",
            container_name="b70-m04-candidate-196k", image=CANDIDATE_IMAGE,
            payload_overrides={"ignore_eos": True})
        candidate_log = (run / "candidate-196k/engine.log").read_text()
        if "B70_M04_SHARED_KV_DISPATCH" not in candidate_log:
            raise RuntimeError("candidate did not dispatch M04")
        control_decode = control["scheduler"]["metric_delta"][
            "vllm:request_decode_time_seconds_sum"]
        candidate_decode = candidate["scheduler"]["metric_delta"][
            "vllm:request_decode_time_seconds_sum"]
        control_prefill = control["scheduler"]["metric_delta"][
            "vllm:request_prefill_time_seconds_sum"]
        candidate_prefill = candidate["scheduler"]["metric_delta"][
            "vllm:request_prefill_time_seconds_sum"]
        assessment = {
            "control": {
                "ttft_s": control["ttft_s"], "decode_s": control_decode,
                "prefill_s": control_prefill,
                "completion_tokens": control["usage"]["completion_tokens"],
                "preemptions": control["scheduler"]["metric_delta"][
                    "vllm:num_preemptions_total"],
                "recomputed_tokens": control["scheduler"][
                    "recomputed_scheduler_tokens"]},
            "candidate": {
                "ttft_s": candidate["ttft_s"], "decode_s": candidate_decode,
                "prefill_s": candidate_prefill,
                "completion_tokens": candidate["usage"]["completion_tokens"],
                "preemptions": candidate["scheduler"]["metric_delta"][
                    "vllm:num_preemptions_total"],
                "recomputed_tokens": candidate["scheduler"][
                    "recomputed_scheduler_tokens"]},
        }
        assessment["control"]["decode_tps"] = (
            (assessment["control"]["completion_tokens"] - 1) /
            control_decode)
        assessment["candidate"]["decode_tps"] = (
            (assessment["candidate"]["completion_tokens"] - 1) /
            candidate_decode)
        assessment["decode_speedup_pct"] = 100 * (
            assessment["candidate"]["decode_tps"] /
            assessment["control"]["decode_tps"] - 1)
        assessment["prefill_change_pct"] = 100 * (
            control_prefill / candidate_prefill - 1)
        assessment["ttft_change_pct"] = 100 * (
            control["ttft_s"] / candidate["ttft_s"] - 1)
        assessment["pass"] = (
            assessment["decode_speedup_pct"] >= 1 and
            assessment["control"]["completion_tokens"] ==
            assessment["candidate"]["completion_tokens"] and
            assessment["candidate"]["preemptions"] <=
            assessment["control"]["preemptions"] and
            assessment["candidate"]["recomputed_tokens"] <=
            assessment["control"]["recomputed_tokens"] and
            assessment["ttft_change_pct"] >= -2)
        state["assessment"] = assessment
        state["status"] = ("qualified-for-promotion" if assessment["pass"]
                           else "rejected")
        state["finished_at_unix"] = time.time()
        save(run / "state.json", state)
        return 0
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        state["finished_at_unix"] = time.time()
        save(run / "state.json", state)
        raise
    finally:
        subprocess.run(["python3", str(REPO / "scripts/restore-production.py")],
                       check=True, timeout=800,
                       env={**os.environ, "B70_RUN_DIR": str(run)})


if __name__ == "__main__":
    raise SystemExit(main())

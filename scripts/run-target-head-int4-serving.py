#!/usr/bin/env python3
"""Screen the repaired target head; always restore production, never promote."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("diag", REPO / "scripts/run-diagnostics.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)
CONTROL = "sha256:aee9857bef1f37c8f0ee136d9f89d7166201212175a8b171d958627706cf1c0b"
NAME = "b70-m06-target-head-arm"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse completed arms with the same pinned images")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.resume:
        state = json.loads((run / "state.json").read_text())
        diag.save(run / f"state-before-resume-{time.time_ns()}.json", state)
        complete = []
        for arm in state["arms"]:
            expected = {8192, 65536, 131072} if arm["index"] < 2 else {8192, 65536}
            if {m["target_context"] for m in arm["measurements"]} == expected:
                complete.append(arm)
            else:
                state.setdefault("prior_attempts", []).append(arm)
        state["arms"] = complete
        state["status"] = "resume-preflight"
        state.pop("error", None)
        state.pop("finished_at", None)
    else:
        run.mkdir(parents=True, exist_ok=False)
        state = {"status": "preflight", "started_at": time.time(), "arms": [],
                 "scope": "ABBA-style candidate/control/control/candidate screen; no production promotion"}

    def save():
        diag.save(run / "state.json", state)

    save()
    lock = Path("/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    production = diag.production_inspect()
    if production["Image"] != CONTROL or int(diag.POWER_CAP.read_text()) != 180000000:
        raise RuntimeError("production image/power changed")
    diag.assert_mtp4(production["Config"]["Cmd"])
    diag.ensure_idle(diag.PRODUCTION_URL)
    candidate = subprocess.check_output(["docker", "image", "inspect", args.candidate,
                                         "--format", "{{.Id}}"], text=True).strip()
    manifest = {"control": CONTROL, "candidate": candidate,
                "patch_sha256": hashlib.sha256((REPO / "benchmarks/experiments/m06-target-head-int4/candidate/patch_target_head.py").read_bytes()).hexdigest()}
    if args.resume:
        if json.loads((run / "manifest.json").read_text()) != manifest:
            raise RuntimeError("resume image or patch differs from original run")
    else:
        diag.save(run / "production-inspect.json", production)
        diag.save(run / "manifest.json", manifest)
    process = None
    try:
        subprocess.run(["systemctl", "--user", "stop", diag.PRODUCTION_SERVICE], check=True, timeout=120)
        for index, label in enumerate(("candidate", "control", "control", "candidate")):
            if any(arm["index"] == index for arm in state["arms"]):
                continue
            phase = run / f"arm-{index}-{label}"
            if phase.exists():
                phase = run / f"arm-{index}-{label}-retry-{time.time_ns()}"
            phase.mkdir()
            image = candidate if label == "candidate" else CONTROL
            row = {"index": index, "label": label, "image": image, "directory": str(phase), "measurements": []}
            state["arms"].append(row)
            state["status"] = f"loading-{index}-{label}"
            save()
            command = diag.engine_command(production, name=NAME, image=image,
                                          evidence=phase, arguments=production["Config"]["Cmd"].copy())
            diag.save(phase / "engine-command.json", command)
            with (phase / "engine.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                diag.wait_ready(process)
                models = diag.api("/v1/models")
                if models["data"][0]["max_model_len"] != 200704:
                    raise RuntimeError("context limit changed")
                text = (phase / "engine.log").read_text()
                if label == "candidate" and text.count("B70_TARGET_LMHEAD_INT4_READY") != 1:
                    raise RuntimeError("target head must be packed exactly once")
                if label == "candidate" and "source_numel 0" not in text:
                    raise RuntimeError("FP16 target head was not freed")
                row["startup_pass"] = True
                save()
                if index < 2:
                    state["status"] = f"quality-{index}-{label}"
                    save()
                    checks = [
                        ["python3", str(REPO / "scripts/check-target-head-quality.py"), "--image", CONTROL, "--output-dir", str(phase / "quality")],
                        ["python3", "/home/sebastian/LocalLLM/Local-AI-B70/qwen38/production/check-vision-tools.py", "--root", diag.EXPERIMENT_URL, "--output", str(phase / "vision-tools.json")],
                        ["python3", str(REPO / "scripts/check-prefix-state.py"), "--output", str(phase / "prefix-state.json")],
                    ]
                    for number, check in enumerate(checks):
                        with (phase / f"quality-{number}.log").open("w") as output:
                            subprocess.run(check, check=True, stdout=output, stderr=subprocess.STDOUT, timeout=1800)
                    row["quality"] = json.loads((phase / "quality/results.json").read_text())
                    # Task-level failures are measured outcomes, including in
                    # the control. They must not silently remove the speed arm.
                    save()
                contexts = (8192, 65536, 131072) if index < 2 else (8192, 65536)
                for context in contexts:
                    for warm in (True, False):
                        state["status"] = f"{'warmup' if warm else 'measure'}-{index}-{label}-{context}"
                        save()
                        output = phase / f"{'warmup' if warm else 'bench'}-{context}"
                        # Different early prefixes prevent APC hits between lengths or warmup.
                        nonce = f"{'warm' if warm else 'measure'}-{context}-target-head-fixed"
                        check = ["python3", str(REPO / "scripts/benchmark.py"), "--root", diag.EXPERIMENT_URL,
                                 "--contexts", str(context), "--output-tokens", "32" if warm else "512",
                                 "--repeats", "1", "--nonce", nonce, "--output-dir", str(output)]
                        with (phase / f"{output.name}.log").open("w") as stream:
                            subprocess.run(check, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=2200)
                        result = json.loads((output / "results.json").read_text())["rows"][0]
                        if result["completion_tokens"] != (32 if warm else 512):
                            raise RuntimeError("incomplete benchmark output")
                        if not warm:
                            row["measurements"].append(result)
                            save()
                diag.stop_container(NAME, process)
                process = None
        summary = []
        for context in (8192, 65536, 131072):
            comparison = {"context": context}
            for label in ("control", "candidate"):
                values = [m for arm in state["arms"] if arm["label"] == label
                          for m in arm["measurements"] if m["target_context"] == context]
                comparison[label] = {"n": len(values), **{key: statistics.median(m[key] for m in values)
                    for key in ("decode_tokens_per_second", "prefill_seconds", "ttft_seconds", "accepted_per_drafted")}}
            comparison["decode_change_pct"] = 100 * (comparison["candidate"]["decode_tokens_per_second"] / comparison["control"]["decode_tokens_per_second"] - 1)
            summary.append(comparison)
        state["summary"] = summary
        state["status"] = "screen-complete-not-promoted"
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        raise
    finally:
        diag.stop_container(NAME, process)
        state["finished_at"] = time.time()
        save()
        subprocess.run(["python3", str(REPO / "scripts/restore-production.py")], check=True,
                       timeout=800, env={**os.environ, "B70_RUN_DIR": str(run)})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run C2 candidate/control on one B70 and restore the candidate service."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "benchmark-results/production-release-v1/heldout-quality-v1"
BASE = "http://127.0.0.1:8081"
CANDIDATE = "sha256:a42cda993bf6492acc39d23e9382e27a17efca4657bea07d80d0c28228a1623a"
CONTROL = "sha256:f0d7bc4ea6040cf4dc8d139b01e0fb348cad80dfaecdaf8d56cd560fb9103559"
CONTROL_NAME = "b70-heldout-control"
CANDIDATE_NAME = "b70-qwen38-vllm"


def save(name: str, value):
    path = ROOT / name
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def status(state: str, detail: str):
    save("status.json", {"state": state, "detail": detail,
                         "updated_unix": time.time()})
    print(json.dumps({"state": state, "detail": detail}), flush=True)


def inspect(name: str):
    result = subprocess.run(["docker", "inspect", name], capture_output=True,
                            text=True, timeout=30)
    return json.loads(result.stdout)[0] if result.returncode == 0 else None


def wait_health(expected_image: str, name: str, timeout=900,
                process: subprocess.Popen | None = None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"{name} exited before readiness: {process.returncode}")
        item = inspect(name)
        if item and item["Image"] != expected_image:
            raise RuntimeError(f"wrong image on {name}: {item['Image']}")
        if item and item["State"]["Status"] == "exited":
            raise RuntimeError(f"{name} exited before readiness")
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=2) as response:
                if response.status == 200 and item:
                    return item
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(3)
    raise TimeoutError(f"{name} not ready")


def assert_idle():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()
    for metric in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        values = [float(line.split()[-1]) for line in lines
                  if line.startswith(metric + "{")]
        if sum(values) != 0:
            raise RuntimeError(f"endpoint busy: {metric}={sum(values)}")


def run_arm(arm: str, image: str, container: str):
    output = ROOT / arm
    command = [sys.executable, str(REPO / "scripts/run-heldout-quality.py"),
               "--arm", arm, "--expected-image-id", image,
               "--container", container, "--output", str(output)]
    save(f"{arm}-command.json", command)
    with (ROOT / f"{arm}.log").open("w") as log:
        result = subprocess.run(command, cwd=REPO, stdout=log,
                                stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"{arm} qualification failed; see {arm}.log")
    return json.loads((output / "summary.json").read_text())


def control_command(candidate: dict):
    args = candidate["Config"]["Cmd"]
    if ("--max-model-len" not in args or args[args.index("--max-model-len") + 1] != "200704"
            or "--max-num-seqs" not in args or args[args.index("--max-num-seqs") + 1] != "4"):
        raise RuntimeError("candidate launch contract changed")
    render_gid = str(Path("/dev/dri/renderD128").stat().st_gid)
    cache = Path.home() / ".cache/b70-heldout-control" / CONTROL.removeprefix("sha256:")
    (cache / "vllm").mkdir(parents=True, exist_ok=True)
    (cache / "triton").mkdir(parents=True, exist_ok=True)
    env = {
        "HF_HUB_OFFLINE": "1", "VLLM_TARGET_DEVICE": "xpu",
        "ZE_FLAT_DEVICE_HIERARCHY": "COMPOSITE", "ZE_AFFINITY_MASK": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "MODEL_ID": "mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16",
        "MODEL_REVISION": "a47b0c6f0d756bc394c4cc629d5b0ded1acc7001",
        "B70_GPTQ_W4A8_PREFILL": "1", "B70_GPTQ_W4A8_MIN_TOKENS": "512",
        "B70_MTP_BF16_DRAFT": "1", "B70_DRAFT_LMHEAD_INT4": "1",
        "B70_DRAFT_MTP_INT4": "1", "B70_DRAFT_VOCAB_ENABLED": "0",
        "B70_DRAFT_VOCAB_PATH": "", "VLLM_XPU_ENABLE_XPU_GRAPH": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": "/opt/b70-runtime",
    }
    command = ["docker", "run", "--rm", "--name", CONTROL_NAME,
               "--device", "/dev/dri", "--group-add", render_gid,
               "-v", "/dev/dri:/dev/dri:ro", "--shm-size", "8g",
               "-p", "127.0.0.1:8081:8000",
               "-v", f"{Path.home() / '.cache/huggingface'}:/root/.cache/huggingface",
               "-v", f"{cache / 'vllm'}:/root/.cache/vllm",
               "-v", f"{cache / 'triton'}:/root/.triton/cache",
               "-v", f"{REPO / 'runtime'}:/opt/b70-runtime:ro"]
    for key, value in env.items():
        command += ["-e", f"{key}={value}"]
    return command + [CONTROL, *args]


def compare():
    candidate_rows = {row["id"]: row for row in map(json.loads, (
        ROOT / "candidate/rows.jsonl").read_text().splitlines())}
    control_rows = {row["id"]: row for row in map(json.loads, (
        ROOT / "control/rows.jsonl").read_text().splitlines())}
    if len(candidate_rows) != 48 or len(control_rows) != 48:
        raise RuntimeError("paired quality rows incomplete")
    if set(candidate_rows) != set(control_rows):
        raise RuntimeError("candidate/control task IDs differ")
    wins, losses, both_fail = [], [], []
    for name, row in candidate_rows.items():
        other = control_rows[name]
        if row["context_id"] != other["context_id"] or row["kind"] != other["kind"]:
            raise RuntimeError(f"task mismatch: {name}")
        if row["metrics"]["preemptions"] or other["metrics"]["preemptions"]:
            raise RuntimeError(f"preemption in held-out task: {name}")
        a, b = row["score"]["passed"], other["score"]["passed"]
        if a and not b:
            wins.append(name)
        elif b and not a:
            losses.append(name)
        elif not a and not b:
            both_fail.append(name)
    candidate_pass = sum(row["score"]["passed"] for row in candidate_rows.values())
    control_pass = sum(row["score"]["passed"] for row in control_rows.values())
    new_critical = [name for name in losses if candidate_rows[name]["critical"]]
    loss_categories = {kind: sum(candidate_rows[name]["kind"] == kind for name in losses)
                       for kind in {row["kind"] for row in candidate_rows.values()}}
    result = {"candidate_passed": candidate_pass, "control_passed": control_pass,
              "wins": wins, "losses": losses, "both_failed": both_fail,
              "new_critical_failures": new_critical,
              "loss_categories": loss_categories,
              "practical_quality_gate_passed": (
                  len(losses) <= 1 and not new_critical
                  and max(loss_categories.values(), default=0) <= 1
                  and candidate_pass >= control_pass - 1
                  and candidate_rows["review-11"]["score"]["passed"])}
    save("paired-summary.json", result)
    return result


def restore_candidate():
    subprocess.run(["docker", "stop", "-t", "30", CONTROL_NAME],
                   capture_output=True, timeout=60)
    if inspect(CANDIDATE_NAME) is None:
        command = ["systemd-run", "--user", "--no-block",
                   "--unit=b70-candidate-qualification-v2",
                   "--description=B70 restored candidate after held-out control",
                   "--property=Type=exec", str(REPO / "scripts/run-server.sh")]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"candidate restore failed: {result.stderr}")
    wait_health(CANDIDATE, CANDIDATE_NAME)


def main():
    if ROOT.exists():
        raise ValueError("paired result directory already exists")
    ROOT.mkdir(parents=True)
    lock_path = Path("/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        candidate = wait_health(CANDIDATE, CANDIDATE_NAME, timeout=30)
        assert_idle()
        save("candidate-inspect.json", {"id": candidate["Id"],
                                        "image_id": candidate["Image"],
                                        "command": candidate["Config"]["Cmd"]})
        if subprocess.check_output(["docker", "image", "inspect", CONTROL,
                                    "--format", "{{.Id}}"], text=True).strip() != CONTROL:
            raise RuntimeError("frozen W4A8 control image missing")
        control_cmd = control_command(candidate)
        save("control-command.json", control_cmd)
        stopped_candidate = False
        paired_complete = False
        control_process = None
        control_log = None
        try:
            status("running", "candidate")
            run_arm("candidate", CANDIDATE, CANDIDATE_NAME)
            status("running", "switch-to-control")
            stopped_candidate = True
            subprocess.run(["systemctl", "--user", "stop",
                            "b70-candidate-qualification-v1.service"],
                           check=True, timeout=180)
            for _ in range(30):
                if inspect(CANDIDATE_NAME) is None:
                    break
                time.sleep(1)
            if inspect(CANDIDATE_NAME) is not None:
                raise RuntimeError("candidate container did not stop")
            control_log = (ROOT / "control-server.log").open("w")
            control_process = subprocess.Popen(control_cmd, cwd=REPO,
                                               stdout=control_log,
                                               stderr=subprocess.STDOUT)
            wait_health(CONTROL, CONTROL_NAME, process=control_process)
            assert_idle()
            status("running", "control")
            run_arm("control", CONTROL, CONTROL_NAME)
            result = compare()
            paired_complete = True
            status("evaluated", "paired score recorded")
            print(json.dumps(result), flush=True)
        except Exception as error:
            status("failed", f"{type(error).__name__}: {error}")
            raise
        finally:
            if stopped_candidate:
                status("restoring", "candidate service")
                restore_candidate()
                status("complete" if paired_complete else "failed",
                       "paired run and candidate restored" if paired_complete
                       else "paired run failed; candidate restored")
            if control_process is not None:
                try:
                    control_process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    control_process.kill()
            if control_log is not None:
                control_log.close()


if __name__ == "__main__":
    main()

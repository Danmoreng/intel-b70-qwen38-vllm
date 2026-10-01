"""Run bounded-memory reference/native arms; optional explicit GPTQ restoration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

REPO = Path(__file__).resolve().parents[3]
ORIGINAL = "hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
GPTQ = "hub/models--mikeinnyc--Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16/snapshots/a47b0c6f0d756bc394c4cc629d5b0ded1acc7001"
PRODUCTION_IMAGE = "local/b70-qwen38-vllm:production-onednn-v2"
EXL_IMAGE = "local/exl3xpu:15ded2f"


def call(args, **kw):
    print(json.dumps({"command": args, "unix": time.time()}), flush=True)
    return subprocess.run(args, check=True, **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(REPO / "benchmark-results/quantization-reference-20261001"))
    p.add_argument("--arms", nargs="+", default=["bf16", "fp16", "gptq", "exl3"])
    p.add_argument("--windows", type=int, default=0)
    p.add_argument("--manifest-name", default="campaign.json")
    p.add_argument("--restore-production", action="store_true")
    p.add_argument("--gptq-snapshot", default=GPTQ,
                   help="Pinned snapshot path relative to HF_HOME")
    p.add_argument("--gptq-arm", default="gptq",
                   help="Distinct result label for an alternative GPTQ checkpoint")
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    home = Path.home()
    hf = home / ".cache/huggingface"
    exl = home / ".cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw"
    assert (out / "panel.json").is_file()
    assert args.gptq_arm.startswith("gptq") and all(
        c.isalnum() or c == "-" for c in args.gptq_arm), args.gptq_arm
    allowed = {"bf16", "fp16", "gptq", "gptq-w4a8", "exl3", "exl3-int8",
               args.gptq_arm, args.gptq_arm + "-w4a8"}
    for arm in args.arms:
        assert arm in allowed, arm
        if (out / arm / "summary.json").exists():
            raise RuntimeError(f"Refusing to overwrite completed arm {arm}")
    snapshot = (hf / args.gptq_snapshot).resolve()
    assert snapshot.is_relative_to(hf.resolve()) and (snapshot / "config.json").is_file()
    release = json.loads((REPO / "config/production_image.json").read_text())
    expected = release["image_id"]
    actual = subprocess.check_output(["docker", "image", "inspect", PRODUCTION_IMAGE, "--format", "{{.Id}}"], text=True).strip()
    assert actual == expected
    metadata = {"started_unix": time.time(), "production_image_id": expected,
                "exl_image_id": subprocess.check_output(["docker", "image", "inspect", EXL_IMAGE, "--format", "{{.Id}}"], text=True).strip(),
                "panel_sha256": hashlib.sha256((out / "panel.json").read_bytes()).hexdigest(),
                "script_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")},
                "arms": args.arms, "completed": [], "restore_production_requested": args.restore_production,
                "gptq_snapshot": args.gptq_snapshot, "gptq_arm": args.gptq_arm}
    manifest = out / args.manifest_name
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    name = "b70-quantization-reference-eval"

    def interrupted(signum, frame):
        raise InterruptedError(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    stopped = False
    try:
        call(["systemctl", "--user", "stop", "b70-qwen38-vllm.service"], timeout=90)
        stopped = True
        for arm in args.arms:
            common = ["docker", "run", "--rm", "--name", name,
                "--user", "1000:1000", "--group-add", str(os.stat("/dev/dri/renderD128").st_gid),
                "--group-add", str(os.stat("/dev/dri/card0").st_gid),
                "--device", "/dev/dri", "--network", "bridge" if arm.startswith("exl3") else "none",
                "--memory", "12g", "--shm-size", "1g",
                "--mount", "type=bind,src=/dev/dri/by-path,dst=/dev/dri/by-path,readonly",
                "--env", "HF_HOME=/cache", "--env", "HF_HUB_OFFLINE=1",
                "--env", "PYTHONPATH=/scripts", "--env", "VLLM_WORKER_MULTIPROC_METHOD=spawn",
                "--env", "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE", "--env", "ZE_AFFINITY_MASK=0",
                "--env", "OMP_NUM_THREADS=4", "--env", f"XDG_CACHE_HOME=/results/runtime-cache/{arm}",
                "--env", f"TRITON_CACHE_DIR=/results/runtime-cache/{arm}/triton",
                "--env", "PYTORCH_ALLOC_CONF=expandable_segments:True",
                "--env", "B70_ONEDNN_PREFILL=1", "--env", "B70_DRAFT_VOCAB_ENABLED=0",
                "--env", f"B70_GPTQ_W4A8_PREFILL={int(arm in {'gptq-w4a8', args.gptq_arm + '-w4a8'})}",
                "--env", "B70_GPTQ_W4A8_MIN_TOKENS=512",
                "--env", "EXL3_ONEDNN_ATTN=0",
                "--env", f"EXL3_INT8_PREFILL={int(arm == 'exl3-int8')}",
                "--mount", f"type=bind,src={hf},dst=/cache,readonly",
                "--mount", f"type=bind,src={exl},dst=/exl3,readonly",
                "--mount", f"type=bind,src={Path(__file__).parent},dst=/scripts,readonly",
                "--mount", f"type=bind,src={out},dst=/results", "--entrypoint", "python"]
            if arm in {"bf16", "fp16"}:
                command = common + [PRODUCTION_IMAGE, "-u", "/scripts/stream_reference.py",
                    "--model", f"/cache/{ORIGINAL}", "--dtype", "bfloat16" if arm == "bf16" else "float16"]
            else:
                is_exl = arm.startswith("exl3")
                command = common + [EXL_IMAGE if is_exl else PRODUCTION_IMAGE, "-u", "/scripts/run_native.py",
                    "--model", "/exl3" if is_exl else f"/cache/{args.gptq_snapshot}",
                    "--quantization", "exl3" if is_exl else "gptq"]
            command += ["--panel", "/results/panel.json", "--out", f"/results/{arm}"]
            if args.windows:
                command += ["--windows", str(args.windows)]
            with (out / f"{arm}.log").open("w") as log:
                call(command, stdout=log, stderr=subprocess.STDOUT)
            metadata["completed"].append(arm)
            manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    except BaseException as exc:
        metadata["error"] = repr(exc)
        raise
    finally:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if stopped and args.restore_production:
            call(["systemctl", "--user", "start", "b70-qwen38-vllm.service"], timeout=180)
            deadline = time.monotonic() + 900
            while True:
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=5) as response:
                        assert response.status == 200
                    break
                except (OSError, AssertionError):
                    if time.monotonic() > deadline:
                        raise RuntimeError("Production health did not recover")
                    time.sleep(2)
            actual = subprocess.check_output(["docker", "inspect", "b70-qwen38-vllm", "--format", "{{.Image}}"], text=True).strip()
            assert actual == expected, "Production restored wrong image"
            metadata["production_restored"] = {"health": 200, "image_id": actual, "unix": time.time()}
        elif stopped:
            call(["systemctl", "--user", "stop", "b70-qwen38-vllm.service"], timeout=90)
            metadata["production_left_offline"] = True
        metadata["finished_unix"] = time.time()
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()

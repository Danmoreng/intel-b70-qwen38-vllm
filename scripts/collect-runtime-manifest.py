#!/usr/bin/env python3
"""Collect a value-free, attributable manifest from a running vLLM container."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


REPO = Path(__file__).resolve().parents[1]
WORKER_PATHS = (
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/xpu_worker.py",
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py",
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py",
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/xpu_model_runner.py",
    "/opt/venv/lib/python3.12/site-packages/b70_attention.py",
    "/opt/b70/tiles.so",
)


def output(command: list[str], *, check: bool = True) -> str:
    result = subprocess.run(command, check=check, text=True, capture_output=True)
    return result.stdout.strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def container_probe(container: str) -> dict[str, object]:
    source = r'''import hashlib, importlib.metadata, json, platform, subprocess
from pathlib import Path
paths = json.loads(__import__("os").environ["B70_MANIFEST_PATHS"])
versions = {}
for name in ("vllm", "vllm-xpu-kernels", "torch", "triton", "pytorch-triton-xpu"):
    try: versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name] = None
files = {}
for item in paths:
    path = Path(item)
    files[item] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
driver_packages = {}
for name in ("intel-igc-core-2", "intel-igc-opencl-2", "intel-ocloc", "intel-opencl-icd", "libze-intel-gpu1", "libigdgmm12"):
    result = subprocess.run(["dpkg-query", "-W", "-f=${Version}", name], text=True, capture_output=True)
    driver_packages[name] = result.stdout if result.returncode == 0 else None
print(json.dumps({"packages": versions, "installed_files_sha256": files,
                  "driver_packages": driver_packages, "container_kernel": platform.release()}))'''
    return json.loads(
        output(
            [
                "docker",
                "exec",
                "-e",
                "B70_MANIFEST_PATHS=" + json.dumps(WORKER_PATHS),
                container,
                "python",
                "-c",
                source,
            ]
        )
    )


def git_identity() -> dict[str, object]:
    revision = output(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    status = output(["git", "-C", str(REPO), "status", "--porcelain=v1"])
    diff = output(["git", "-C", str(REPO), "diff", "--binary", "HEAD"])
    untracked = []
    for line in status.splitlines():
        if line.startswith("?? "):
            path = REPO / line[3:]
            if path.is_file():
                untracked.append(
                    {"path": line[3:], "sha256": sha256(path.read_bytes())}
                )
    return {
        "revision": revision,
        "dirty": bool(status),
        "tracked_diff_sha256": sha256(diff.encode()),
        "untracked_files": untracked,
    }


def power_policy() -> dict[str, object]:
    caps = list(Path("/sys/bus/pci/devices/0000:03:00.0/hwmon").glob("*/power1_cap"))
    return {
        "pci_device": "0000:03:00.0",
        "power_cap_uw": int(caps[0].read_text()) if len(caps) == 1 else None,
        "power_cap_source_count": len(caps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="qwen38-vllm-production")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inspect = json.loads(output(["docker", "inspect", args.container]))[0]
    config = inspect["Config"]
    selected_env = {}
    for row in config.get("Env") or []:
        name, _, value = row.partition("=")
        if name.startswith(("B70_", "VLLM_", "ZE_", "PYTORCH_")) or name in {
            "HF_HUB_OFFLINE",
            "PYTHONPATH",
        }:
            selected_env[name] = value
    manifest = {
        "schema": 1,
        "captured_at_unix": time.time(),
        "control": "q128-mtp4-full-180w",
        "production_container": args.container,
        "container_image_reference": config["Image"],
        "container_image_id": inspect["Image"],
        "container_command": (config.get("Entrypoint") or []) + (config.get("Cmd") or []),
        "selected_environment": selected_env,
        "mount_destinations": sorted(item["Destination"] for item in inspect["Mounts"]),
        "runtime": container_probe(args.container),
        "repository": git_identity(),
        "power_policy": power_policy(),
        "model_identity": {
            "model": "mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16",
            "revision": "a47b0c6f0d756bc394c4cc629d5b0ded1acc7001",
            "tokenizer_revision": "same pinned model revision",
            "high_precision_source_lineage": "unresolved; not inferred from the live config",
        },
        "quantization": {
            "target": "GPTQ symmetric INT4 group-size 128, W4A16 runtime",
            "draft": "BF16 MTP parameters with INT4 LM head and five INT4 linears",
            "draft_vocabulary": "full 248320-token vocabulary",
            "kv_cache": "FP8",
        },
        "storage_identity_audit": {
            "status": "pending in-process object inspection",
            "known": "packing code reads head.weight to create owned INT4 tensors",
            "not_claimed": "the wrapper alone does not prove source storage was released",
            "weight_values_recorded": False,
        },
        "rollback": "systemctl --user start qwen38.service",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

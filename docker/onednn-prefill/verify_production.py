#!/usr/bin/env python3
"""Fail before loading the model when release bytes or policy are incompatible."""

import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path("/opt/b70")
POLICY = ROOT / "production_policy.json"
XPU_LINEAR = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/"
    "kernels/linear/mixed_precision/xpu.py")
EXPECTED_LINEAR = "58b0b4fc0972d1bfd2bdf294910fa6bd5841d0c511d1de41ca88d9164b884945"
EXPECTED_ATTENTION = "b49fab77ace6d6e0744f82286a5a9585f164e6e05fcf7d01c2f5c88ec018edc9"
EXPECTED_BASE_ATTENTION = "d84d857e1f13ad690dca9e8697f56ec89df1b7351142690d9fa0d5919eae2205"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(f"B70 production verification failed: {message}")


def main():
    arguments = sys.argv[1:]
    require(arguments == ["--files-only"] or
            (arguments and arguments[0] == "--serve" and len(arguments) > 1),
            "expected --files-only or --serve MODEL [args]")
    policy = json.loads(POLICY.read_text())
    expected_policy = (ROOT / "production_policy.sha256").read_text().split()[0]
    require(digest(POLICY) == expected_policy, "policy file hash")
    require(os.environ.get("B70_POLICY_SHA256") == expected_policy,
            "built policy hash")
    files = {
        ROOT / "libdnnl.so.3": policy["attention"]["onednn"]["required_onednn_library_sha256"],
        ROOT / "native_sdpa.so": policy["attention"]["onednn"]["required_native_operator_sha256"],
        ROOT / "tiles.so": policy["attention"]["q128_library_sha256"],
        ROOT / "m04.so": policy["attention"]["m04_library_sha256"],
        XPU_LINEAR: EXPECTED_LINEAR,
        Path("/opt/venv/lib/python3.12/site-packages/b70_attention.py"):
            EXPECTED_ATTENTION,
        Path("/opt/venv/lib/python3.12/site-packages/b70_attention_base.py"):
            EXPECTED_BASE_ATTENTION,
    }
    for path, expected in files.items():
        require(path.is_file() and digest(path) == expected, str(path))
    if arguments == ["--files-only"]:
        return
    expected_env = {
        "B70_GPTQ_W4A8_PREFILL": "1",
        "B70_GPTQ_W4A8_MIN_TOKENS": "512",
        "B70_ONEDNN_PREFILL": "1",
        "B70_ONEDNN_MIXED_ROUTE": "1",
        "B70_ONEDNN_MIN_KV": "16384",
        "B70_ONEDNN_MAX_KV": "196608",
        "B70_DRAFT_VOCAB_ENABLED": "0",
        "B70_DRAFT_LMHEAD_INT4": "1",
        "B70_DRAFT_MTP_INT4": "1",
    }
    for name, expected in expected_env.items():
        require(os.environ.get(name) == expected, f"{name} must equal {expected}")
    for name in ("B70_ONEDNN_PROFILE", "B70_ONEDNN_SHORT_CHUNK_ONLY",
                 "B70_ONEDNN_FINAL_CHUNK_ONLY", "B70_ONEDNN_VALIDATE",
                 "B70_ONEDNN_VALIDATE_FP32", "B70_ONEDNN_MIXED_VALIDATE",
                 "B70_QUALIFICATION_ROUTE_OFF", "B70_QUALIFICATION_CONTROL"):
        require(os.environ.get(name) in (None, "0"), f"{name} unsupported")
    require(os.environ.get("MODEL_REVISION") == policy["model"]["revision"],
            "model revision")
    require(os.environ.get("MODEL_ID") == policy["model"]["id"], "model ID")
    print("B70_PRODUCTION_VERIFIED", {
        "policy_sha256": expected_policy,
        "file_sha256": {str(path): digest(path) for path in files},
    }, flush=True)
    os.execvp("vllm", ["vllm", "serve", *arguments[1:]])


if __name__ == "__main__":
    main()

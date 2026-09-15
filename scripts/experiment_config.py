#!/usr/bin/env python3
"""Build validated, non-production vLLM experiment arguments.

The module has no service-management side effects.  Its CLI only prints a
machine-readable dry-run manifest so experiments can be reviewed and hashed
before a container is started by a separate orchestrator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from typing import Mapping


@dataclass(frozen=True)
class ExperimentConfig:
    image: str = "local/qwen38-b70-vllm:q128-196k-20260914"
    model: str = "mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
    revision: str = "a47b0c6f0d756bc394c4cc629d5b0ded1acc7001"
    served_name: str = "Qwen3.8-27B"
    quantization: str = "gptq"
    dtype: str = "float16"
    context_size: int = 200_704
    gpu_memory_utilization: float = 0.93
    kv_cache_dtype: str = "fp8"
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 4_096
    speculative_tokens: int = 4
    prefix_caching: bool = True
    mamba_cache_mode: str = "align"
    power_limit_w: int = 180

    def validate(self) -> None:
        if self.context_size < 1:
            raise ValueError("context_size must be positive")
        if self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")
        if self.speculative_tokens < 0:
            raise ValueError("speculative_tokens cannot be negative")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if not self.quantization.strip():
            raise ValueError("quantization cannot be empty")

    def server_args(self) -> list[str]:
        self.validate()
        args = [
            "serve",
            self.model,
            "--revision",
            self.revision,
            "--quantization",
            self.quantization,
            "--dtype",
            self.dtype,
            "--max-model-len",
            str(self.context_size),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--kv-cache-dtype",
            self.kv_cache_dtype,
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--max-num-batched-tokens",
            str(self.max_num_batched_tokens),
            "--enable-prefix-caching" if self.prefix_caching else "--no-enable-prefix-caching",
            "--mamba-cache-mode",
            self.mamba_cache_mode,
            "--served-model-name",
            self.served_name,
        ]
        if self.speculative_tokens:
            args += [
                "--speculative-config",
                json.dumps(
                    {"method": "mtp", "num_speculative_tokens": self.speculative_tokens},
                    separators=(",", ":"),
                ),
            ]
        args += [
            "--middleware",
            "request_defaults.ChatDefaults",
            "--reasoning-config",
            '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}',
            "--override-generation-config",
            '{"max_new_tokens":16384}',
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_xml",
            "--reasoning-parser",
            "qwen3",
            "--limit-mm-per-prompt",
            '{"image":1,"video":0}',
            "--mm-processor-kwargs",
            '{"max_pixels":4194304}',
        ]
        return args

    def manifest(self) -> dict[str, object]:
        return {
            "schema": 1,
            "purpose": "review-only experiment dry run; no service mutation",
            "config": asdict(self),
            "server_args": self.server_args(),
            "environment": runtime_environment(),
        }


def runtime_environment() -> dict[str, str]:
    return {
        "HF_HUB_OFFLINE": "1",
        "VLLM_TARGET_DEVICE": "xpu",
        "ZE_FLAT_DEVICE_HIERARCHY": "COMPOSITE",
        "ZE_AFFINITY_MASK": "0",
        "B70_MTP_BF16_DRAFT": "1",
        "B70_DRAFT_LMHEAD_INT4": "1",
        "B70_DRAFT_MTP_INT4": "1",
        "B70_DRAFT_VOCAB_ENABLED": "0",
        "B70_DRAFT_VOCAB_PATH": "",
        "VLLM_XPU_ENABLE_XPU_GRAPH": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": "/opt/qwen-production",
    }


def from_environment(env: Mapping[str, str] | None = None) -> ExperimentConfig:
    values = os.environ if env is None else env

    def integer(name: str, default: int) -> int:
        try:
            return int(values.get(name, str(default)))
        except ValueError as error:
            raise ValueError(f"{name} must be an integer") from error

    def floating(name: str, default: float) -> float:
        try:
            return float(values.get(name, str(default)))
        except ValueError as error:
            raise ValueError(f"{name} must be a number") from error

    defaults = ExperimentConfig()
    config = ExperimentConfig(
        image=values.get("VLLM_IMAGE", defaults.image),
        model=values.get("MODEL_ID", defaults.model),
        revision=values.get("MODEL_REVISION", defaults.revision),
        served_name=values.get("SERVED_MODEL_NAME", defaults.served_name),
        quantization=values.get("QUANTIZATION", defaults.quantization),
        dtype=values.get("DTYPE", defaults.dtype),
        context_size=integer("CONTEXT_SIZE", defaults.context_size),
        gpu_memory_utilization=floating(
            "GPU_MEMORY_UTILIZATION", defaults.gpu_memory_utilization
        ),
        kv_cache_dtype=values.get("KV_CACHE_DTYPE", defaults.kv_cache_dtype),
        max_num_seqs=integer("MAX_NUM_SEQS", defaults.max_num_seqs),
        max_num_batched_tokens=integer(
            "MAX_NUM_BATCHED_TOKENS", defaults.max_num_batched_tokens
        ),
        speculative_tokens=integer("SPECULATIVE_TOKENS", defaults.speculative_tokens),
        prefix_caching=values.get("PREFIX_CACHING", "1") == "1",
        mamba_cache_mode=values.get("MAMBA_CACHE_MODE", defaults.mamba_cache_mode),
        power_limit_w=integer("POWER_LIMIT_W", defaults.power_limit_w),
    )
    config.validate()
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the validated JSON manifest (the CLI never launches a service)",
    )
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("--dry-run is required; launching is delegated to an orchestrator")
    print(json.dumps(from_environment().manifest(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Add an opt-in profiler start immediately before XPU graph capture."""
from pathlib import Path


path = Path("/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/xpu_worker.py")
source = path.read_text()
anchor = "    def profile(self, is_start: bool = True, profile_prefix: str | None = None):\n"
addition = '''    def compile_or_warm_up_model(self) -> None:
        # Diagnostic-only: graph replay internals are visible to the XPU torch
        # profiler only when it is active while those graphs are captured.
        if os.getenv("B70_PROFILE_BEFORE_CAPTURE") == "1":
            logger.info("Starting diagnostic profiler before XPU graph capture")
            self.profile(is_start=True, profile_prefix="before_capture")
        return super().compile_or_warm_up_model()

'''
if addition in source:
    raise SystemExit("pre-capture profiler patch is already present")
if source.count(anchor) != 1:
    raise SystemExit(f"expected exactly one XPUWorker profile anchor, got {source.count(anchor)}")
path.write_text(source.replace(anchor, addition + anchor))

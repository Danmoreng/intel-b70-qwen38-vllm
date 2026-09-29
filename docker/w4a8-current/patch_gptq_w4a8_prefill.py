#!/usr/bin/env python3
"""Port the GPTQ prefill-only W4A8 experiment to pinned vLLM 0.30.0 XPU."""

from __future__ import annotations

import hashlib
from pathlib import Path

import vllm


EXPECTED_SHA256 = "6c4fc2a961dfc94a75aa39c91f76d53621c8903291820c72946ca889ef2c4816"
MARKER = "B70_GPTQ_W4A8_PREFILL"


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one anchor, found {source.count(old)}: {old[:80]!r}")
    return source.replace(old, new, 1)


def main() -> None:
    path = Path(vllm.__file__).parent / "model_executor/kernels/linear/mixed_precision/xpu.py"
    source = path.read_text()
    if MARKER in source:
        raise RuntimeError(f"patch already present in {path}")
    actual = hashlib.sha256(source.encode()).hexdigest()
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"vLLM XPU source changed: {actual}; expected {EXPECTED_SHA256}")

    source = replace_once(source, "\n\nimport torch\n", "\n\nimport os\n\nimport torch\n")
    source = replace_once(
        source,
        '''        if c.weight_type != scalar_types.int4:
            return (
                False,
                f"XPUW4A8Int requires int4 weights, got {c.weight_type}",
            )''',
        '''        if c.weight_type != scalar_types.int4 and not (
            c.weight_type == scalar_types.uint4b8
            and os.environ.get("B70_GPTQ_W4A8_PREFILL") == "1"
        ):
            return (
                False,
                f"XPUW4A8Int requires int4 weights or env-gated symmetric GPTQ, got {c.weight_type}",
            )''',
    )
    source = replace_once(
        source,
        '''    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_scale.data = layer.weight_scale.data.t().contiguous()

        device = layer.weight_packed.device''',
        '''    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.config.weight_type == scalar_types.uint4b8:
            # Preserve the exact current GPTQ packing, scales, and zero-point setup.
            XPUwNa16LinearKernel.process_weights_after_loading(self, layer)
            self._b70_w4a8_min_tokens = int(os.environ.get("B70_GPTQ_W4A8_MIN_TOKENS", "512"))
            if self._b70_w4a8_min_tokens < 1:
                raise ValueError("B70_GPTQ_W4A8_MIN_TOKENS must be positive")
            logger.info_once(
                "B70 GPTQ W4A8 prefill enabled at >=%d rows; smaller matrices use W4A16",
                self._b70_w4a8_min_tokens,
            )
            return
        layer.weight_scale.data = layer.weight_scale.data.t().contiguous()

        device = layer.weight_packed.device''',
    )
    source = replace_once(
        source,
        '''        reshaped_x = x.reshape(-1, x.shape[-1])  # [M, K]
        from vllm._xpu_ops import xpu_ops as ops

        # TODO: static and asymmetric quantization case''',
        '''        reshaped_x = x.reshape(-1, x.shape[-1])  # [M, K]
        if self.config.weight_type == scalar_types.uint4b8:
            if reshaped_x.shape[0] < self._b70_w4a8_min_tokens:
                return XPUwNa16LinearKernel.apply_weights(self, layer, x, bias)
            from vllm._xpu_ops import xpu_ops as ops

            w_q, w_s, w_zp = self._get_weight_params(layer)
            quant_x, x_scale, x_zero = ops.dynamic_per_token_int8_quant_ref(
                reshaped_x, True, 8
            )
            out = torch.ops._xpu_C.int4_gemm_w4a8(
                quant_x, x_scale, x_zero, w_q.t(), w_s, w_zp,
                self.config.group_size, None, bias,
            )
            return out.to(x.dtype)

        from vllm._xpu_ops import xpu_ops as ops

        # TODO: static and asymmetric quantization case''',
    )
    compile(source, str(path), "exec")
    path.write_text(source)
    print(f"patched {path}; sha256={hashlib.sha256(source.encode()).hexdigest()}")


if __name__ == "__main__":
    main()

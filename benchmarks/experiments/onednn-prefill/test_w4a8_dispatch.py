#!/usr/bin/env python3
"""Exercise the installed GPTQ dispatch, not just the two underlying operators."""

import json
from types import SimpleNamespace

import torch
from vllm.model_executor.kernels.linear.mixed_precision.xpu import XPUW4A8IntLinearKernel
from vllm.scalar_type import scalar_types
from vllm._xpu_ops import xpu_ops
from vllm_xpu_kernels import _xpu_C  # noqa: F401


def main():
    torch.manual_seed(70)
    size = 5120
    packed = torch.randint(-(2**31), 2**31 - 1, (size, size // 8),
                           device="xpu", dtype=torch.int32)
    scales = (torch.rand((size // 128, size), device="xpu",
                         dtype=torch.float16) * .015 + .0005).contiguous()
    zero = torch.tensor([8], device="xpu", dtype=torch.int8)
    kernel = SimpleNamespace(
        config=SimpleNamespace(weight_type=scalar_types.uint4b8, group_size=128),
        _b70_w4a8_min_tokens=512,
        _get_weight_params=lambda layer: (packed, scales, zero),
    )
    for rows in (511, 512, 513):
        x = torch.randn((rows, size), device="xpu", dtype=torch.float16)
        dispatched = XPUW4A8IntLinearKernel.apply_weights(kernel, None, x)
        w4a16 = torch.ops._xpu_C.int4_gemm_w4a16(
            x, packed.t(), None, scales, zero, 128, None)
        quant, factor, origin = xpu_ops.dynamic_per_token_int8_quant_ref(x, True, 8)
        w4a8 = torch.ops._xpu_C.int4_gemm_w4a8(
            quant, factor, origin, packed.t(), scales, zero, 128, None, None)
        expected = w4a16 if rows == 511 else w4a8
        delta = (dispatched.float() - expected.float()).abs()
        other = w4a8 if rows == 511 else w4a16
        output = {
            "rows": rows, "selected": "w4a16" if rows == 511 else "w4a8",
            "dispatch_exact": bool(torch.equal(dispatched, expected)),
            "other_operator_max_abs": float((dispatched.float() - other.float()).abs().max().item()),
            "max_abs": float(delta.max().item()),
        }
        print(json.dumps(output), flush=True)
        if not output["dispatch_exact"]:
            raise RuntimeError(f"incorrect W4A8 dispatch at {rows} rows")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare the pinned GPTQ W4A8 operator with W4A16 near row dispatch."""

import json

import torch
from vllm._xpu_ops import xpu_ops
from vllm_xpu_kernels import _xpu_C  # noqa: F401  -- registers torch ops


def main():
    torch.manual_seed(70)
    input_size, output_size = 5120, 5120
    packed = torch.randint(-(2**31), 2**31 - 1,
                           (output_size, input_size // 8), device="xpu",
                           dtype=torch.int32).t()
    scales = (torch.rand((input_size // 128, output_size), device="xpu",
                         dtype=torch.float16) * .015 + .0005).contiguous()
    zero = torch.tensor([8], device="xpu", dtype=torch.int8)
    records = []
    for rows in (511, 512, 513):
        for profile in ("typical", "zero", "outlier"):
            x = torch.randn((rows, input_size), device="xpu", dtype=torch.float16)
            if profile == "zero":
                x.zero_()
            elif profile == "outlier":
                x[0, 0] = 100.0
            reference = torch.ops._xpu_C.int4_gemm_w4a16(
                x, packed, None, scales, zero, 128, None)
            quant, factor, origin = xpu_ops.dynamic_per_token_int8_quant_ref(x, True, 8)
            candidate = torch.ops._xpu_C.int4_gemm_w4a8(
                quant, factor, origin, packed, scales, zero, 128, None, None)
            torch.xpu.synchronize()
            delta = (reference.float() - candidate.float()).abs()
            row = {
                "rows": rows, "profile": profile,
                "finite": bool(torch.isfinite(candidate).all().item()),
                "rms": float(delta.square().mean().sqrt().item()),
                "relative_rms": float((delta.square().mean().sqrt() /
                                        reference.float().square().mean().sqrt()).item())
                    if profile != "zero" else 0.0,
                "max_abs": float(delta.max().item()),
                "zero_exact": bool(torch.count_nonzero(candidate).item() == 0)
                    if profile == "zero" else None,
            }
            records.append(row)
            print(json.dumps(row), flush=True)
    if not all(row["finite"] and (row["zero_exact"] is not False) for row in records):
        raise RuntimeError("W4A8 boundary operator failed finite/zero check")


if __name__ == "__main__":
    main()

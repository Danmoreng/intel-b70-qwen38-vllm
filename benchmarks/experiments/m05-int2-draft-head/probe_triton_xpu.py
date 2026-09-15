#!/usr/bin/env python3
"""Minimal capability probe for the Intel Triton backend used by M05."""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x, y, out, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(out + offsets, tl.load(x + offsets) + tl.load(y + offsets))


x = torch.arange(16, device="xpu", dtype=torch.float16)
y = torch.ones_like(x)
out = torch.empty_like(x)
add_kernel[(1,)](x, y, out, n=16)
torch.xpu.synchronize()
torch.testing.assert_close(out.cpu(), (x + y).cpu())
print({"status": "PASS", "triton": triton.__version__, "device": str(x.device)})

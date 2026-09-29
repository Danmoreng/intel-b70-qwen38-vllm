#!/usr/bin/env python3
"""Exercise more than 32 exact Q/L signatures and reuse an evicted signature."""

import json

import torch


def main():
    torch.ops.load_library("/opt/b70/native_sdpa.so")
    q = torch.randn((6, 256, 256), device="xpu", dtype=torch.float16) * .1
    k = torch.randn((1, 4128, 256), device="xpu", dtype=torch.float16) * .1
    v = torch.randn_like(k) * .1
    out = torch.empty_like(q)
    divisor = torch.tensor([16], device="xpu", dtype=torch.float16)
    negative_inf = torch.tensor([float("-inf")], device="xpu", dtype=torch.float32)

    def execute(length):
        torch.ops.b70_sdpa_probe.forward(
            q, k[:, :length], v[:, :length], out, divisor, negative_inf)

    execute(4096)
    original = out.clone()
    for length in range(4097, 4129):
        execute(length)
    execute(4096)
    torch.xpu.synchronize()
    result = {"shapes": 33, "revisited": 4096,
              "exact_replay": bool(torch.equal(original, out)),
              "finite": bool(torch.isfinite(out).all().item())}
    print(json.dumps(result), flush=True)
    if not (result["exact_replay"] and result["finite"]):
        raise RuntimeError("cache eviction changed output")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare the fused FP8 page gather with the Torch conversion oracle."""

import json

import torch


def main():
    torch.ops.load_library("/probe/native_sdpa.so")
    bits = torch.arange(256, dtype=torch.uint8)
    bits[127] = 0
    bits[255] = 0
    raw = bits.view(1, 1, 1, 1, 256).expand(3, 1664, 4, 2, 256).clone()
    cache = raw.view(torch.float8_e4m3fn).to("xpu")
    key, value = cache[:, :, :, 0], cache[:, :, :, 1]
    pages = torch.tensor([2, 0, 1], device="xpu", dtype=torch.int32)
    rows = []
    for length in (1663, 1664, 1665, 3329):
        for source, head, factor in ((key, 2, .75), (value, 3, 1.25)):
            scale = torch.tensor([factor], device="xpu", dtype=torch.float32)
            output = torch.empty((length, 256), device="xpu", dtype=torch.float16)
            torch.ops.b70_sdpa_probe.gather_dequant(source, pages, scale, output, head)
            reference = (source[:, :, head].index_select(0, pages.long())
                         .flatten(0, 1)[:length].to(torch.float16) * scale).to(torch.float16)
            torch.xpu.synchronize()
            difference = (output.float() - reference.float()).abs()
            row = {"length": length, "head": head, "scale": factor,
                   "exact": bool(torch.equal(output, reference)),
                   "max_abs": float(difference.max().item()),
                   "mismatch_count": int((output != reference).sum().item())}
            rows.append(row)
            print(json.dumps(row), flush=True)
    if not all(row["exact"] for row in rows):
        raise RuntimeError("native gather differs from Torch oracle")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exercise final-image mixed routing, exact boundaries, empty rows and capture."""

import json
from unittest.mock import patch

import torch

import b70_attention as route


def inputs():
    torch.manual_seed(70)
    q = torch.randn((258, 24, 256), device="xpu", dtype=torch.float16) * .02
    cache = (torch.randn((10, 1664, 4, 2, 256), device="xpu",
                         dtype=torch.float16) * .02).to(torch.float8_e4m3fn)
    k, v = cache[:, :, :, 0, :], cache[:, :, :, 1, :]
    table = torch.arange(10, device="xpu", dtype=torch.int32).repeat(2, 1)
    return dict(q=q, k=k, v=v, cu_seqlens_q=torch.tensor([0, 256, 258],
                device="xpu", dtype=torch.int32),
                seqused_k=torch.tensor([16384, 1024], device="xpu", dtype=torch.int32),
                block_table=table, max_seqlen_q=256, max_seqlen_k=16384,
                causal=True, softmax_scale=.0625, window_size=(-1, -1),
                k_descale=torch.tensor([1.0], device="xpu", dtype=torch.float32),
                v_descale=torch.tensor([1.0], device="xpu", dtype=torch.float32))


def main():
    d = inputs()
    first = route._subcall(d, [0, 256, 258], [16384, 1024], 0, None)
    assert route.eligible(first)
    for length, expected in [(16383, False), (16384, True),
                             (196608, True), (196609, False)]:
        single = dict(first)
        single["max_seqlen_k"] = length
        single["block_table"] = torch.zeros(
            (1, (length + 1663) // 1664), device="xpu", dtype=torch.int32)
        assert route.eligible(single) is expected, (length, expected)
    with patch.object(torch.xpu, "is_current_stream_capturing", return_value=True):
        assert not route.eligible(first)
    mixed = route.flash_attn_varlen_func(**d)
    reference = route.fallback(**d)
    torch.xpu.synchronize()
    delta = (mixed.float() - reference.float()).abs()
    result = {"mixed_allclose": bool(torch.allclose(mixed, reference,
                                                    rtol=.01, atol=.002)),
              "mixed_max_abs": float(delta.max().item()),
              "mixed_relative_l2": float((delta.norm() /
                                           reference.float().norm()).item())}
    print(json.dumps(result), flush=True)
    assert result["mixed_allclose"]
    padded = dict(d)
    padded["cu_seqlens_q"] = torch.tensor([0, 256, 256, 258],
                                           device="xpu", dtype=torch.int32)
    padded["seqused_k"] = torch.tensor([16384, 0, 1024],
                                        device="xpu", dtype=torch.int32)
    padded["block_table"] = torch.cat([d["block_table"][:1],
                                       d["block_table"][:1],
                                       d["block_table"][1:]], dim=0)
    padded_result = route.flash_attn_varlen_func(**padded)
    assert torch.allclose(padded_result, mixed, rtol=.01, atol=.002)
    reordered = dict(d)
    reordered["q"] = torch.cat([d["q"][256:], d["q"][:256]], dim=0)
    reordered["cu_seqlens_q"] = torch.tensor([0, 2, 258], device="xpu",
                                             dtype=torch.int32)
    reordered["seqused_k"] = torch.tensor([1024, 16384], device="xpu",
                                           dtype=torch.int32)
    reordered["block_table"] = d["block_table"].flip(0).contiguous()
    ordered_result = route.flash_attn_varlen_func(**reordered)
    assert torch.allclose(ordered_result,
                          torch.cat([mixed[256:], mixed[:256]]),
                          rtol=.01, atol=.002)
    print(json.dumps({"empty_row_matches": True,
                      "reordered_rows_match": True, "capture_rejected": True,
                      "route_counts": route._COUNTERS}), flush=True)


if __name__ == "__main__":
    main()

"""Exact prefill / bounded decode metadata, prepared once by the V2 builder.

The CPU sequence length is exact ONLY on genuine prefill rows. Decode rows
retain their native GPU lengths. CPU query boundaries are trusted only with
adaptive verification disabled on the source-guarded single-rank V2 runner.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Segment:
    first: int
    last: int
    start: int
    stop: int
    maximum_query: int
    maximum_kv: int
    prefill: bool
    cu_query: object = None


@dataclass(frozen=True)
class HostPlan:
    requests: int
    tokens: int
    segments: tuple


def host_segments(bounds, lengths, prefilling, tokens, *, fixed_partition):
    """Pure host contract; None means use the unmodified native whole batch."""
    if not fixed_partition or type(tokens) is not int or tokens <= 0:
        return None
    n = len(lengths)
    if (n <= 0 or len(bounds) != n + 1 or len(prefilling) != n or
            any(type(x) is not int for x in bounds + lengths) or
            any(type(x) is not bool for x in prefilling) or
            bounds[0] != 0 or bounds[-1] != tokens):
        return None
    queries = [bounds[i+1] - bounds[i] for i in range(n)]
    if any(q <= 0 or length < q for q, length in zip(queries, lengths)):
        return None
    long = [flag and q >= 64 and length >= 4096
            for flag, q, length in zip(prefilling, queries, lengths)]
    if not any(long):
        return None
    segments = []
    i = 0
    while i < n:
        j = i + 1
        # Long prefills have individual exact KV lengths; native work can be
        # batched in contiguous runs, independent of request ordering.
        if not long[i]:
            while j < n and not long[j]:
                j += 1
        segments.append(Segment(i, j, bounds[i], bounds[j],
                                max(queries[i:j]), max(lengths[i:j]), long[i]))
        i = j
    return HostPlan(n, tokens, tuple(segments))


def prepare(common, *, fixed_partition):
    if not fixed_partition or common.max_query_len < 64:
        return None
    tensors = [getattr(common, name, None) for name in
               ('query_start_loc_cpu', 'seq_lens_cpu_upper_bound', 'is_prefilling')]
    if any(not isinstance(t, torch.Tensor) or t.device.type != 'cpu' or t.ndim != 1
           for t in tensors):
        return None
    bounds, lengths, flags = tensors
    n = common.num_reqs
    if (type(n) is not int or n <= 0 or bounds.numel() < n+1 or
            lengths.numel() < n or flags.numel() < n or flags.dtype != torch.bool or
            bounds.dtype not in (torch.int32, torch.int64) or
            lengths.dtype not in (torch.int32, torch.int64)):
        return None
    bounds = bounds[:n+1].tolist()
    plan = host_segments(bounds, lengths[:n].tolist(), flags[:n].tolist(),
                         common.num_actual_tokens, fixed_partition=True)
    if plan is None:
        return None
    result = []
    for s in plan.segments:
        cu = None
        if not s.prefill:
            if s.start == 0:
                cu = common.query_start_loc[s.first:s.last+1]
            else:
                # One asynchronous metadata copy per native run per step,
                # retained by the metadata object, never per attention layer.
                host = torch.tensor([x-s.start for x in bounds[s.first:s.last+1]],
                                    dtype=torch.int32)
                if common.query_start_loc.device.type != 'cpu':
                    host = host.pin_memory()
                cu = host.to(common.query_start_loc.device, non_blocking=True)
        result.append(Segment(s.first, s.last, s.start, s.stop,
                              s.maximum_query, s.maximum_kv, s.prefill, cu))
    return HostPlan(plan.requests, plan.tokens, tuple(result))

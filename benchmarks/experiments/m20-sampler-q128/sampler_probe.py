#!/usr/bin/env python3
"""Compare current MRV2 sampling with the published fused XPU operator."""

import json
from pathlib import Path
import statistics
import time

import torch
import vllm._xpu_ops  # noqa: F401  registers torch.ops.vllm.*
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample


OUT = Path('/evidence/sampler.json')
VOCAB = 248320
TOP_K = 20
TOP_P = 0.95


def timed(fn):
    start = torch.xpu.Event(enable_timing=True)
    stop = torch.xpu.Event(enable_timing=True)
    torch.xpu.synchronize()
    before = time.perf_counter()
    start.record()
    result = fn()
    stop.record()
    torch.xpu.synchronize()
    return result, start.elapsed_time(stop), (time.perf_counter() - before) * 1000


def one_batch(batch):
    torch.manual_seed(38 + batch)
    logits = torch.randn((batch, VOCAB), device='xpu', dtype=torch.float32)
    scratch = torch.empty_like(logits)
    k = torch.full((batch,), TOP_K, device='xpu', dtype=torch.int32)
    p = torch.full((batch,), TOP_P, device='xpu', dtype=torch.float32)
    mapping = torch.arange(batch, device='xpu', dtype=torch.int64)
    temperature = torch.ones(batch, device='xpu', dtype=torch.float32)
    seeds = torch.arange(38, 38 + batch, device='xpu', dtype=torch.int64)
    positions = torch.ones(batch, device='xpu', dtype=torch.int64)
    sampled = torch.empty(batch, device='xpu', dtype=torch.int64)
    generator = torch.xpu.default_generators[0]

    def native():
        scratch.copy_(logits)
        filtered = apply_top_k_top_p(scratch, k, p)
        return gumbel_sample(filtered, mapping, temperature, seeds, positions,
                             apply_temperature=False, is_drafting=False)

    def fused():
        scratch.copy_(logits)
        state = generator.get_state()
        seed, offset = state.view(torch.int64).tolist()
        rng = torch.tensor([seed, offset], dtype=torch.int64, device='cpu')
        torch.ops.vllm.xpu_topk_topp_sampler(
            sampled, None, scratch, k.to(torch.int64), p, 'raw_logprobs', rng
        )
        state.view(torch.int64)[1] = (offset + scratch.numel() + 3) // 4 * 4
        generator.set_state(state)
        return sampled

    for _ in range(4):
        native()
        fused()
    torch.xpu.synchronize()
    allowed = [set(row.tolist()) for row in logits.topk(TOP_K, dim=-1).indices.cpu()]
    times = {'native': {'device_ms': [], 'wall_ms': []},
             'fused': {'device_ms': [], 'wall_ms': []}}
    outputs = {'native': [], 'fused': []}
    for pair in range(8):
        order = ('native', 'fused', 'fused', 'native') if pair % 2 == 0 else (
            'fused', 'native', 'native', 'fused')
        for arm in order:
            result, device_ms, wall_ms = timed(native if arm == 'native' else fused)
            ids = result.cpu().tolist()
            if any(token not in allowed[i] for i, token in enumerate(ids)):
                raise RuntimeError(f'{arm} sampled outside top-{TOP_K}: {ids}')
            outputs[arm].append(ids)
            times[arm]['device_ms'].append(device_ms)
            times[arm]['wall_ms'].append(wall_ms)
    medians = {arm: {kind: statistics.median(vals) for kind, vals in parts.items()}
               for arm, parts in times.items()}
    return {'batch': batch, 'vocab': VOCAB, 'top_k': TOP_K, 'top_p': TOP_P,
            'pairs': 8, 'sample_in_top_k': True, 'medians_ms': medians,
            'raw_timings_ms': times,
            'native_unique_tokens_per_row': [len(set(row[i] for row in outputs['native']))
                                             for i in range(batch)],
            'fused_unique_tokens_per_row': [len(set(row[i] for row in outputs['fused']))
                                            for i in range(batch)]}


def main():
    if not hasattr(torch.ops._xpu_C, 'topk_topp_sampler'):
        raise RuntimeError('published wheel has no fused sampler operator')
    rows = [one_batch(batch) for batch in (1, 2, 4)]
    result = {'torch': torch.__version__, 'wheel': '0.1.15.4',
              'runner': 'MRV2 current native versus PR 57277 fused operator',
              'explicit_request_seeds_supported_by_pr': False, 'rows': rows}
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    for row in rows:
        print(json.dumps({'batch': row['batch'], 'medians_ms': row['medians_ms'],
                          'sample_in_top_k': row['sample_in_top_k']}), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Recompute cold-cell checks and medians from the published counter deltas."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent


def main():
    total = 0
    for name, expected in [('diagnosis-cells.jsonl', 16),
                           ('original-profile-cells.jsonl', 24),
                           ('production-180w.jsonl', 8)]:
        cells = [json.loads(line) for line in (ROOT / name).read_text().splitlines()]
        if len(cells) != expected:
            raise RuntimeError(f'{name}: expected {expected} cells, got {len(cells)}')
        groups = defaultdict(list)
        for c in cells:
            d = c['metrics_delta']
            usage = c['last_chunk_extra']['usage']
            checks = [c['completion_tokens'] == 512,
                      usage['prompt_tokens_details']['cached_tokens'] == 0,
                      usage['completion_tokens_details']['reasoning_tokens'] == 0,
                      d.get('vllm:prefix_cache_hits_total', 0) == 0,
                      d.get('vllm:external_prefix_cache_hits_total', 0) == 0,
                      d.get('vllm:num_preemptions_total', 0) == 0,
                      d.get('vllm:request_success_total') == 1,
                      d.get('vllm:generation_tokens_total') == 512]
            server_s = c['last_chunk_extra']['metrics']['time_to_first_token_ms'] / 1000
            checks.append(abs(c['ttft_s'] - server_s) < 0.3)
            if not all(checks):
                raise RuntimeError(f'Invalid cell in {name}: {c.get("arm")}/{c["rep"]}/{c["depth"]}')
            groups[c.get('arm', 'production-LBNHC'), c['power_cap_w'], c['kind'], c['depth']].append(c)
        print(name)
        for key, rows in groups.items():
            print(f'  {key}: n={len(rows)}, TTFT={median(r["ttft_s"] for r in rows):.3f} s, '
                  f'decode={median(r["decode_tok_s"] for r in rows):.2f} tok/s')
        total += len(cells)
    print(f'{total}/{total} cold cells passed independent counter/timing checks')


if __name__ == '__main__':
    main()

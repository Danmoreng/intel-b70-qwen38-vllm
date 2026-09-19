#!/usr/bin/env python3
"""Verify the published production-only aggregate from content-free requests."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
summary = json.loads((ROOT / 'summary.json').read_text())
raw = (ROOT / 'requests.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == summary['requests_sha256']
rows = json.loads(raw)
assert len(rows) == 133 and [r['index'] for r in rows] == list(range(133))
assert all(r['status'] == 'ok' and r['http_status'] == 200 for r in rows)


def check(records, expected):
    def total(key):
        return sum(r[key] for r in records)

    values = {
        'requests': len(records),
        'valid_requests': len(records),
        'actual_context_min': min(r['prompt_tokens'] for r in records),
        'actual_context_max': max(r['prompt_tokens'] for r in records),
        'prompt_tokens': total('prompt_tokens'),
        'generated_tokens': total('generated_tokens'),
        'computed_tokens': total('computed_tokens'),
        'cached_tokens': total('prefix_hit_tokens'),
        'prefill_seconds': total('native_prefill_s'),
        'decode_seconds': total('native_decode_s'),
        'prefill_compute_tps': total('computed_tokens') / total('native_prefill_s'),
        'decode_tps': (total('generated_tokens') - len(records)) / total('native_decode_s'),
        'prefix_hit_rate': total('prefix_hit_tokens') / total('prefix_query_tokens'),
        'mtp_acceptance': total('accepted_tokens') / total('drafted_tokens'),
    }
    for key, value in values.items():
        assert math.isclose(value, expected[key], rel_tol=1e-10, abs_tol=1e-8), (key, value, expected[key])


check(rows, summary['overall'])
for band in summary['bands']:
    low, high = map(int, band['band'].split('-'))
    check([r for r in rows if low <= r['prompt_tokens'] < high], band)
assert sum(b['requests'] for b in summary['bands']) == len(rows)
assert summary['overall']['computed_tokens'] + summary['overall']['cached_tokens'] == summary['overall']['prompt_tokens']
print('Verified 133 requests, overall metrics and all 15 context bands.')

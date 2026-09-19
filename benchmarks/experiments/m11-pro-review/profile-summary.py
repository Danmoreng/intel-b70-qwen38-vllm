"""Compact evidence from already recorded traces; never launches GPU work."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
for arm in ('control', 'candidate'):
    trace = next((args.run / f'block-0-{arm}/trace').glob('*.gz'))
    events = json.loads(gzip.decompress(trace.read_bytes()))['traceEvents']
    kernels = collections.defaultdict(lambda: {'count': 0, 'duration_us': 0})
    for event in events:
        if event.get('cat') == 'kernel':
            kernels[event['name']]['count'] += 1
            kernels[event['name']]['duration_us'] += event.get('dur', 0)
    total = sum(k['duration_us'] for k in kernels.values())
    # Include both native FillFunctor and compiler-generated zeros. Never count
    # a name containing "prefill" as an initialization just because of "fill".
    fills = {name: data for name, data in kernels.items()
             if 'FillFunctor<' in name or 'fused_zeros_' in name or name == '_zero_kv_blocks_kernel'}
    result = {
        'trace_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(),
        'scope': 'instrumented 8K/32-output request; prefill device kernels visible, decode graph interiors opaque',
        'warning': 'Kernel duration sums are neither critical-path time nor a serving speedup prediction.',
        'kernel_count': sum(k['count'] for k in kernels.values()),
        'kernel_duration_sum_ms': total / 1000,
        'all_visible_initialization_ms': sum(k['duration_us'] for k in fills.values()) / 1000,
        'all_visible_initialization_pct_of_kernel_sum': 100 * sum(k['duration_us'] for k in fills.values()) / total,
        'initialization_kernels': fills,
        'norm_transform_kernels': {n: k for n, k in kernels.items() if '_to_copy_add' in n},
        'e01_fused_calls': kernels.get('_fused_qk_rmsnorm_rope_gate_kernel', {'count': 0, 'duration_us': 0}),
    }
    (args.run / f'{arm}-profile-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(arm, json.dumps({k: v for k, v in result.items() if k not in ('initialization_kernels', 'norm_transform_kernels')}))

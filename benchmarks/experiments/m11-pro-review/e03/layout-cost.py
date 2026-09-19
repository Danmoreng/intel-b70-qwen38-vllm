"""Measure E03's removable layout region on the production M04 binary.

The elision arm deliberately returns packed output: it is an upper-bound
diagnostic, NOT a correct candidate and cannot qualify a production change.
Both arms use the same Q/K/V, allocations and fixed production split policy.
"""
import json
import hashlib
import random
import statistics
from pathlib import Path
import time

import torch
from vllm_xpu_kernels import _vllm_fa2_C  # register dependencies

OUT = Path('/work')
assert hashlib.sha256(Path('/opt/b70/m04.so').read_bytes()).hexdigest() == '784916abd42b614794becac634a6a154f854c73d6dfcb1a50cd939b563020c6a'
torch.ops.load_library('/opt/b70/m04.so')
torch.set_grad_enabled(False)
torch.manual_seed(19091903)
PAGE, HQ, HKV, D = 1664, 24, 4, 256
CONTEXTS = (8192, 16384, 65536, 196608)
MAX_K = 200704  # Observed production FULL-graph M04 dispatch bound.
SPLITS = {2: 32, 3: 8, 4: 16, 5: 16}
pages = (MAX_K + PAGE - 1) // PAGE
base = torch.randn((pages, PAGE, HKV, 2, D), device='xpu',
                   dtype=torch.float16).to(torch.float8_e4m3fn)
k, v = base[:, :, :, 0, :], base[:, :, :, 1, :]
assert k.stride() == v.stride() == (3407872, 2048, 512, 1)
physical = torch.randperm(pages, device='xpu').int()
ks = torch.tensor([.75], device='xpu', dtype=torch.float32)
vs = torch.tensor([1.25], device='xpu', dtype=torch.float32)


def measure(fn, repeats=10):
    torch.xpu.synchronize()
    a, b = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    start = time.perf_counter()
    a.record()
    for _ in range(repeats):
        fn()
    b.record()
    b.synchronize()
    return {'device_ms': a.elapsed_time(b) / repeats,
            'wall_ms': (time.perf_counter() - start) * 1000 / repeats}


rows = []
for m, splits in SPLITS.items():
    for context in CONTEXTS:
        q = torch.randn((m, HQ, D), device='xpu', dtype=torch.float16)
        bt = physical[None, :].contiguous()
        cq = torch.tensor([0, m], device='xpu', dtype=torch.int32)
        used = torch.tensor([context], device='xpu', dtype=torch.int32)
        packed_out = torch.empty((1, m * HQ, D), device='xpu', dtype=q.dtype)
        final = torch.empty_like(q)
        temp = torch.empty((1, m * HQ * splits, D), device='xpu', dtype=q.dtype)
        sums = torch.empty((1, m * HQ, splits), device='xpu', dtype=torch.float32)
        maxima = torch.empty_like(sums)

        def native():
            pq = q.view(m, HKV, 6, D).permute(1, 0, 2, 3).reshape(1, m * HQ, D).contiguous()
            torch.ops.b70_ops.shared_kv_verify_out(
                pq, k, v, bt, cq, used, ks, vs, packed_out, temp, sums, maxima,
                MAX_K, splits, 8)
            return packed_out

        def layout():
            result = packed_out.view(HKV, m, 6, D).permute(1, 0, 2, 3).reshape(m, HQ, D).contiguous()
            final.copy_(result)
            return final

        def full():
            native()
            return layout()

        functions = {'production_region': full, 'elide_output_layout_INVALID': native,
                     'layout_only': layout}
        for fn in functions.values():
            for _ in range(8):
                fn()
        # Prove the measured layout reproduces the intended token/head map.
        expected = torch.stack([torch.stack([packed_out[0, (h // 6) * m * 6 + row * 6 + h % 6]
                                 for h in range(HQ)]) for row in range(m)])
        assert torch.equal(layout(), expected)
        assert bool(expected.isfinite().all())
        for mode in ('eager', 'graph'):
            graphs = {}
            if mode == 'graph':
                for name, fn in functions.items():
                    graph = torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph):
                        value = fn()
                    graphs[name] = (graph, value)
                timed = {name: pair[0].replay for name, pair in graphs.items()}
            else:
                timed = functions
            samples = {name: [] for name in timed}
            rng = random.Random(190919 + m + context)
            for _ in range(21):
                order = list(timed)
                rng.shuffle(order)
                for name in order:
                    samples[name].append(measure(timed[name]))
            medians = {name: {metric: statistics.median(s[metric] for s in data)
                             for metric in ('device_ms', 'wall_ms')}
                       for name, data in samples.items()}
            full_ms = medians['production_region']['device_ms']
            saved_ms = full_ms - medians['elide_output_layout_INVALID']['device_ms']
            row = {'m': m, 'context': context, 'max_seqlen_k': MAX_K, 'splits': splits, 'mode': mode,
                   'medians': medians, 'removable_ms_upper_estimate': saved_ms,
                   'removable_pct_upper_estimate': 100 * saved_ms / full_ms,
                   'samples': samples}
            rows.append(row)
            (OUT / 'layout-cost.json').write_text(json.dumps({
                'scope': 'operator-region diagnostic on synthetic Q/K/V; elision is NOT a correct implementation',
                'source_binary': '/opt/b70/m04.so', 'rows': rows}, indent=2) + '\n')
            print(json.dumps({key: value for key, value in row.items() if key != 'samples'}), flush=True)

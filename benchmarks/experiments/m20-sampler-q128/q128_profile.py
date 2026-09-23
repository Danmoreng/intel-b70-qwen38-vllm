#!/usr/bin/env python3
"""Profile the deployed Q128 attention operator at 64K and 192K."""

import json
from pathlib import Path
import subprocess
import statistics
import time

import torch
from torch.profiler import profile, ProfilerActivity
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func


OUT = Path('/evidence/q128.json')
BLOCK = 1664
Q = 6656
LENGTHS = (65536, 196608)


def timed(fn):
    start = torch.xpu.Event(enable_timing=True)
    stop = torch.xpu.Event(enable_timing=True)
    start.record()
    result = fn()
    stop.record()
    torch.xpu.synchronize()
    return result, start.elapsed_time(stop)


def main():
    torch.manual_seed(38)
    torch.ops.load_library('/opt/b70/tiles.so')
    blocks = (max(LENGTHS) + BLOCK - 1) // BLOCK
    q = torch.randn((Q, 24, 256), device='xpu', dtype=torch.float16)
    kv = torch.empty((blocks, BLOCK, 4, 2, 256), device='xpu', dtype=torch.float8_e4m3fn)
    for offset in range(0, blocks, 8):
        count = min(8, blocks - offset)
        chunk = torch.randn((count, BLOCK, 4, 2, 256), device='xpu', dtype=torch.float16)
        kv[offset:offset + count].copy_((chunk * 60).to(torch.float8_e4m3fn))
    k, v = kv[:, :, :, 0, :], kv[:, :, :, 1, :]
    assert k.stride() == (3407872, 2048, 512, 1)
    table = torch.arange(blocks, device='xpu', dtype=torch.int32).reshape(1, -1)
    cu_q = torch.tensor([0, Q], device='xpu', dtype=torch.int32)
    used = torch.tensor([0], device='xpu', dtype=torch.int32)
    descale = torch.full((1,), .01, device='xpu', dtype=torch.float32)
    rows = []
    for length in LENGTHS:
        used.fill_(length)
        torch.xpu.synchronize()

        def q128():
            return torch.ops.b70_tiles.forward(q, k, v, table, cu_q, used,
                                               descale, descale, length, 1)

        def native():
            return flash_attn_varlen_func(
                q, k, v, max_seqlen_q=Q, cu_seqlens_q=cu_q,
                max_seqlen_k=length, seqused_k=used, k_descale=descale,
                v_descale=descale, block_table=table, causal=True,
                softmax_scale=.0625)

        for _ in range(3):
            q128(); native()
        torch.xpu.synchronize()
        output = q128().cpu()
        reference = native().cpu()
        delta = (output.float() - reference.float()).abs()
        correctness = {'exact': bool(torch.equal(output, reference)),
                       'max_abs': float(delta.max().item()),
                       'allclose': bool(torch.allclose(output, reference, rtol=.01, atol=.002))}
        if not correctness['allclose']:
            raise RuntimeError(f'Q128 mismatch at {length}: {correctness}')
        samples = {'q128': [], 'native': []}
        for pair in range(4):
            order = ('q128', 'native', 'native', 'q128') if pair % 2 == 0 else (
                'native', 'q128', 'q128', 'native')
            for arm in order:
                _, elapsed = timed(q128 if arm == 'q128' else native)
                samples[arm].append(elapsed)
        trace_path = Path(f'/evidence/q128-{length}-trace.json')
        try:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                         record_shapes=True) as prof:
                for _ in range(3):
                    q128()
                torch.xpu.synchronize()
            prof.export_chrome_trace(str(trace_path))
            trace_status = 'captured'
        except Exception as exc:
            trace_status = f'unavailable: {type(exc).__name__}: {exc}'
        # Sample the otherwise idle card while only Q128 dispatches execute.
        # These are device-level counters, not per-stage softmax counters.
        monitor = subprocess.Popen(
            ['xpu-smi', 'stats', '-d', '0', '-e', '-j', '--samples', '50',
             '--interval', '100'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True)
        time.sleep(.3)
        for _ in range(24 if length == LENGTHS[0] else 12):
            q128()
        torch.xpu.synchronize()
        try:
            stats_text, error_text = monitor.communicate(timeout=20)
            stats = json.loads(stats_text) if monitor.returncode == 0 else {
                'error': error_text, 'returncode': monitor.returncode}
        except Exception as exc:
            monitor.kill()
            monitor.communicate()
            stats = {'error': f'{type(exc).__name__}: {exc}'}
        (Path('/evidence') / f'q128-{length}-xpu-smi.json').write_text(
            json.dumps(stats, indent=2) + '\n')
        tile = stats.get('memory', {}).get('bandwidth_percent', {}).get('tile_0', {})
        row = {'kv_tokens': length, 'q_tokens': Q, 'kv_stride': list(k.stride()),
               'correctness': correctness, 'timings_ms': samples,
               'median_ms': {name: statistics.median(values)
                             for name, values in samples.items()},
               'trace': trace_status, 'trace_file': trace_path.name,
               'xpu_smi_bandwidth_percent_avg': tile.get('avg'),
               'xpu_smi_file': f'q128-{length}-xpu-smi.json'}
        rows.append(row)
        print(json.dumps({'kv_tokens': length, 'median_ms': row['median_ms'],
                          'trace': trace_status, 'correctness': correctness,
                          'bandwidth_percent_avg': tile.get('avg')}), flush=True)
    OUT.write_text(json.dumps({'torch': torch.__version__, 'rows': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()

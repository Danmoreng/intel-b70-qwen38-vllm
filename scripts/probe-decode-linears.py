"""Replay identical real operands through original/current W4A16 wrappers."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
from vllm.model_executor.kernels.linear.mixed_precision import xpu as linear
from vllm_xpu_kernels import _xpu_C
from vllm.scalar_type import scalar_types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--operands', type=Path, required=True)
    parser.add_argument('--patched', action='store_true')
    parser.add_argument('--load-sdpa', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.load_sdpa:
        torch.ops.load_library('/opt/b70/native_sdpa.so')
    cases = []
    for path in sorted(args.operands.glob('m*.pt')):
        data = torch.load(path, map_location='cpu', weights_only=True)
        tensors = {name: value.to('xpu') if isinstance(value, torch.Tensor) else value
                   for name, value in data.items()}
        proxy = SimpleNamespace(config=SimpleNamespace(weight_type=scalar_types.uint4b8,
                    group_size=data['group_size']), _b70_w4a8_min_tokens=512,
                    _get_weight_params=lambda layer, t=tensors: (t['weight'].t(), t['scales'], t['zero']))
        cls = linear.XPUW4A8IntLinearKernel if args.patched else linear.XPUwNa16LinearKernel
        fn = lambda cls=cls, proxy=proxy, t=tensors: cls.apply_weights(proxy, None, t['input'], t['bias'])
        output = fn()
        torch.xpu.synchronize()
        if not torch.equal(output.cpu(), data['expected']):
            raise RuntimeError(f'bitwise output mismatch: {path}')
        for _ in range(5):
            fn()
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            graph_output = fn()
        graph.replay()
        torch.xpu.synchronize()
        if not torch.equal(graph_output.cpu(), data['expected']):
            raise RuntimeError(f'graph output mismatch: {path}')
        cases.append({'file': path.name, 'shape': data['shape'],
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'bitwise_equal': True, 'weight_stride': list(tensors['weight'].stride()),
                      'fn': fn, 'graph': graph, 'tensors': tensors,
                      'eager_event_ms': [], 'graph_event_ms': [],
                      'eager_wall_ms': [], 'graph_wall_ms': []})
    if not cases:
        raise RuntimeError('no real operand cases')
    for repeat in range(31):
        for case in cases[::1 if repeat % 2 == 0 else -1]:
            for mode in ('eager', 'graph'):
                torch.xpu.synchronize()
                start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
                wall = time.perf_counter()
                start.record()
                if mode == 'graph':
                    case['graph'].replay()
                else:
                    case['fn']()
                end.record()
                end.synchronize()
                case[mode + '_event_ms'].append(start.elapsed_time(end))
                case[mode + '_wall_ms'].append(1000 * (time.perf_counter() - wall))
    results = []
    for case in cases:
        row = {key: value for key, value in case.items() if key not in ('fn', 'graph', 'tensors')}
        row['medians'] = {key: statistics.median(row[key]) for key in
                          ('eager_event_ms', 'graph_event_ms', 'eager_wall_ms', 'graph_wall_ms')}
        results.append(row)
    args.output.write_text(json.dumps({'patched_wrapper': args.patched,
        'sdpa_loaded': args.load_sdpa, 'onednn_version': torch.ops._xpu_C.get_onednn_version(),
        'scope': 'fixed real linear work, rotating shapes; not a full MTP-cycle replay',
        'rows': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()

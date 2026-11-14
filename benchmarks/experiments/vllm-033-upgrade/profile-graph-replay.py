#!/usr/bin/env python3
"""Standalone XPU graph/profiler regression, without vLLM, Triton or weights."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'torch': torch.__version__,
              'torch_git_revision': torch.version.git_version, 'cases': [],
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save():
        (args.out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')

    def function(x):
        return (x * 1.25 + .5).tanh()

    x = torch.randn((128, 34816), device='xpu', dtype=torch.float16)

    def capture(rows):
        gx = x[:rows].clone()
        for _ in range(2):
            function(gx)
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            y = function(gx)
        return graph, gx, y

    def check(label, graph, gx, y):
        graph.replay()
        expected = function(gx)
        equal = bool(torch.equal(y, expected))
        report['cases'].append({'label': label, 'rows': gx.shape[0],
                                'bit_identical_to_eager': equal})
        save()
        assert equal, report['cases'][-1]

    try:
        for phase in ('before-profile', 'after-profile'):
            for rows in (1, 2, 3, 4, 8, 16, 32, 64, 128):
                graph, gx, y = capture(rows)
                check(phase, graph, gx, y)
                gx.mul_(.75)
                check(phase + '-mutated', graph, gx, y)
                del graph, gx, y
            if phase == 'before-profile':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.XPU]):
                    for rows in (128, 1, 20, 128, 129):
                        function(x[:rows])
                    torch.xpu.synchronize()

        graph, gx, y = capture(2)
        check('before-replay-profile', graph, gx, y)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.XPU]) as trace:
            for _ in range(3):
                graph.replay()
            torch.xpu.synchronize()
        path = args.out / 'graph-replay.trace.json'
        trace.export_chrome_trace(str(path))
        events = json.loads(path.read_text())['traceEvents']
        kernels = [event for event in events if event.get('cat') == 'kernel']
        report.update(kernel_events=len(kernels),
                      categories=dict(Counter(event.get('cat') for event in events)),
                      trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        gx.mul_(.75)
        check('after-replay-profile-mutated', graph, gx, y)
        assert len(kernels) >= 9, 'Expected three real kernels for each of three graph replays'
        del graph, gx, y
        for rows in (1, 2, 4):
            graph, gx, y = capture(rows)
            check('after-replay-profile-new-graph', graph, gx, y)
            del graph, gx, y
        report['status'] = 'PASS_GRAPH_NUMERICS_AND_REPLAY_KERNEL_PROFILING'
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc))
        raise
    finally:
        save()
    print(report['status'], 'GPU kernel events:', report['kernel_events'], flush=True)


if __name__ == '__main__':
    main()

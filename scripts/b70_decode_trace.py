"""Offline-only graph/step trace. Device snapshots are read after requests finish."""
import dataclasses
import functools
import json
import os
import sys
from pathlib import Path
import threading
import time


def install():
    import torch
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    root = Path('/evidence')
    pending = []
    state = {'component': 'idle', 'step': 0}

    def inventory(runner):
        seen = set()
        rows = []
        def visit(value, name, depth=0):
            if id(value) in seen or depth > 9:
                return
            seen.add(id(value))
            if isinstance(value, torch.Tensor):
                rows.append({'path': name, 'shape': list(value.shape),
                             'dtype': str(value.dtype), 'device': str(value.device),
                             'bytes': value.numel() * value.element_size()})
            elif isinstance(value, dict):
                for key, child in value.items():
                    visit(child, name + '[' + repr(key) + ']', depth + 1)
            elif isinstance(value, (tuple, list)):
                for key, child in enumerate(value):
                    visit(child, name + '[' + str(key) + ']', depth + 1)
            elif not isinstance(value, torch.nn.Module) and type(value).__module__.startswith('vllm') and hasattr(value, '__dict__'):
                for key, child in vars(value).items():
                    if 'config' not in key and 'model' != key:
                        visit(child, name + '.' + key, depth + 1)
        for name in ('req_states', 'input_buffers', 'block_tables', 'model_state',
                     'sampler', 'speculator', 'cudagraph_manager', 'draft_tokens_handler', 'attn_groups'):
            visit(getattr(runner, name), name)
        for name, module in runner.vllm_config.compilation_config.static_forward_context.items():
            if hasattr(module, 'kv_cache'):
                visit(module.kv_cache, 'kv:' + name)
        (root / f'runner-inventory-{os.getpid()}.json').write_text(json.dumps(rows, indent=2))

    def active():
        return (root / 'trace-active').exists() and len(pending) < 10000

    def wrap(cls, method, component):
        original = getattr(cls, method)

        @functools.wraps(original)
        def call(self, *args, **kwargs):
            if not active():
                return original(self, *args, **kwargs)
            if method == 'execute_model' and not (root / f'runner-inventory-{os.getpid()}.json').exists():
                inventory(self)
            previous = state['component']
            state['component'] = component
            if method == 'execute_model':
                state['step'] += 1
            row = {'kind': method, 'component': component, 'step': state['step'],
                   'host_start_ns': time.monotonic_ns()}
            if method == 'execute_model':
                schedule = args[0] if args else kwargs['scheduler_output']
                row['scheduled_query_lengths'] = list(schedule.num_scheduled_tokens.values())
                row['actual_requests'] = len(schedule.num_scheduled_tokens)
                row['actual_query_rows'] = schedule.total_num_scheduled_tokens
            start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
            start.record()
            try:
                with torch.profiler.record_function('b70_decode:' + component):
                    result = original(self, *args, **kwargs)
                end.record()
                row['host_end_ns'] = time.monotonic_ns()
                pending.append((row, start, end, None))
                return result
            finally:
                state['component'] = previous

        setattr(cls, method, call)

    for cls, method, component in (
        (GPUModelRunner, 'execute_model', 'target_and_prepare'),
        (GPUModelRunner, 'sample_tokens', 'sample_commit_draft'),
        (AutoRegressiveSpeculator, 'propose', 'draft'),
        (RejectionSampler, '__call__', 'rejection_sampler'),
    ):
        wrap(cls, method, component)

    original_graph = CUDAGraphWrapper.__call__

    def graph(self, *args, **kwargs):
        if not active() or not is_forward_context_available():
            return original_graph(self, *args, **kwargs)
        context = get_forward_context()
        descriptor = context.batch_descriptor
        entry = self.concrete_cudagraph_entries.get(descriptor)
        mode = str(context.cudagraph_runtime_mode)
        replay = entry is not None and entry.cudagraph is not None and context.cudagraph_runtime_mode == self.runtime_mode
        row = {'kind': 'graph', 'step': state['step'], 'component': state['component'],
               'mode': mode, 'wrapper_mode': str(self.runtime_mode), 'replay': replay,
               'replay_id': id(entry.cudagraph) if replay else None,
               'descriptor': dataclasses.asdict(descriptor) if descriptor else None,
               'input_shapes': [list(x.shape) for x in args if isinstance(x, torch.Tensor)],
               'host_start_ns': time.monotonic_ns()}
        start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
        start.record()
        result = original_graph(self, *args, **kwargs)
        end.record()
        row['host_end_ns'] = time.monotonic_ns()
        pending.append((row, start, end, None))
        return result

    CUDAGraphWrapper.__call__ = graph
    original_prepare = GPUModelRunner.prepare_attn

    def prepare(self, input_batch):
        result = original_prepare(self, input_batch)
        if active():
            snapshots = {name: getattr(input_batch, name).detach().clone()
                         for name in ('seq_lens', 'query_start_loc', 'positions', 'input_ids')}
            row = {'kind': 'attention_metadata', 'step': state['step'],
                   'actual_requests': input_batch.num_reqs,
                   'actual_query_rows': input_batch.num_tokens,
                   'padded_query_rows': input_batch.num_tokens_after_padding,
                   'scheduled_query_lengths': input_batch.num_scheduled_tokens.tolist(),
                   'length_source': 'V2 device metadata snapshot, read after wave'}
            pending.append((row, None, None, snapshots))
        return result

    GPUModelRunner.prepare_attn = prepare
    original_fullgraph = CudaGraphManager.run_fullgraph

    def fullgraph(self, desc):
        if not active():
            return original_fullgraph(self, desc)
        row = {'kind': 'graph', 'step': state['step'], 'component': state['component'],
               'mode': str(desc.cg_mode), 'replay': True,
               'replay_id': id(self.graphs[desc]),
               'descriptor': dataclasses.asdict(desc),
               'host_start_ns': time.monotonic_ns()}
        # Enums in the dataclass are represented without changing their meaning.
        row['descriptor']['cg_mode'] = str(desc.cg_mode)
        start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
        start.record()
        result = original_fullgraph(self, desc)
        end.record()
        row['host_end_ns'] = time.monotonic_ns()
        pending.append((row, start, end, None))
        return result

    CudaGraphManager.run_fullgraph = fullgraph
    (root / f'trace-installed-{os.getpid()}').touch()

    def flush():
        while True:
            if (root / 'trace-flush').exists() and pending:
                records = pending[:]
                pending.clear()
                torch.xpu.synchronize()
                with (root / f'step-trace-{os.getpid()}.jsonl').open('a') as handle:
                    for row, start, end, snapshots in records:
                        if start is not None:
                            row['stream_elapsed_ms'] = start.elapsed_time(end)
                        if snapshots:
                            row.update({name: tensor.cpu().tolist() for name, tensor in snapshots.items()})
                        handle.write(json.dumps(row) + '\n')
                (root / f'trace-flushed-{os.getpid()}').touch()
            time.sleep(.2)

    threading.Thread(target=flush, daemon=True).start()


if os.environ.get('B70_DECODE_TRACE') == '1' and not any('cpuinfo' in arg for arg in sys.argv):
    install()

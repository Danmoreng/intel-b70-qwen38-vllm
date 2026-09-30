"""Offline V2 cycle snapshots: restore mutable state outside every timed replay.

This hook is never mounted in production. It replays execute_model + sample_tokens
from one real scheduler output, and freezes the sampled commit and target-to-draft
inputs AFTER doing their actual computation. Those copies are included in timing
for every arm. Captured KV/GDN blocks, request state, RNG, buffers and pool indices
are restored before each replay; repeated output/state signatures must agree.
"""
import copy
import dataclasses
import enum
import functools
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def install():
    import numpy as np
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator

    root = Path('/evidence')
    fixtures = Path('/cycle-fixtures')
    state = {'replaying': False, 'pending': None, 'seen': set(), 'canonical': None}
    names = ('req_states', 'input_buffers', 'block_tables', 'model_state', 'sampler',
             'speculator', 'cudagraph_manager', 'draft_tokens_handler', 'attn_groups')

    def primitive(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return True
        if isinstance(value, (list, tuple)):
            return all(primitive(x) for x in value)
        if isinstance(value, dict):
            return all(primitive(k) and primitive(v) for k, v in value.items())
        return False

    def registry(runner, extra_indices=None):
        seen = set()
        tensors, cpu, references, kv = {}, {}, {}, {}
        def visit(value, path, setter=None, depth=0):
            if setter:
                references[path] = (setter, value)
            if depth > 14:
                raise RuntimeError('snapshot traversal depth exceeded: ' + repr(path))
            if isinstance(value, torch.Tensor):
                tensors[path] = value
                if setter:
                    references[path] = (setter, value)
                return
            if isinstance(value, np.ndarray):
                cpu[path] = ('array', value, value.copy())
                return
            if primitive(value):
                if setter:
                    cpu[path] = ('value', setter, copy.deepcopy(value))
                return
            if isinstance(value, enum.Enum) or (dataclasses.is_dataclass(value) and getattr(type(value), '__dataclass_params__').frozen):
                return
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(child, path + (key,), lambda x, v=value, k=key: v.__setitem__(k, x), depth + 1)
            elif isinstance(value, (tuple, list)):
                for key, child in enumerate(value):
                    setter = (lambda x, v=value, k=key: v.__setitem__(k, x)) if isinstance(value, list) else None
                    visit(child, path + (key,), setter, depth + 1)
            elif not isinstance(value, torch.nn.Module) and type(value).__module__.startswith('vllm') and hasattr(value, '__dict__'):
                for key, child in vars(value).items():
                    # Pointers are process-specific. Config, modules, graph objects,
                    # functions, streams and immutable model weights are borrowed.
                    if 'config' in key or 'ptr' in key or key in ('model', '_uva_bufs') or (key == 'pool' and 'Graph' in type(value).__name__):
                        continue
                    visit(child, path + (key,), lambda x, v=value, k=key: setattr(v, k, x), depth + 1)
        for name in names:
            visit(getattr(runner, name), (name,))
        for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            table = runner.block_tables.block_tables[group_id].gpu.cpu()
            lengths = runner.block_tables.num_blocks.np[group_id]
            used = {0}
            for request in runner.req_states.req_id_to_index.values():
                used.update(table[request, :int(lengths[request])].tolist())
            indices = sorted(used)
            for name in group.layer_names:
                module = runner.vllm_config.compilation_config.static_forward_context[name]
                cache = module.kv_cache
                entries = cache if isinstance(cache, (list, tuple)) else (cache,)
                for i, tensor in enumerate(entries):
                    if not isinstance(tensor, torch.Tensor):
                        raise RuntimeError('unsupported KV container: ' + name)
                    # Flash KV: [2, blocks, ...], GDN/conv: [blocks, ...].
                    dim = 1 if tensor.ndim == 5 and tensor.shape[0] == 2 else 0
                    selected_indices = sorted(set(indices) | set((extra_indices or {}).get((name, i), [])))
                    if max(selected_indices) >= tensor.shape[dim]:
                        raise RuntimeError('KV block index outside tensor: ' + name)
                    kv[(name, i)] = (tensor, dim, selected_indices)
        return tensors, cpu, references, kv

    def capture(runner, extra_indices=None):
        tensors, cpu, refs, kv = registry(runner, extra_indices)
        snapshot = {'tensors': {path: t.detach().cpu().clone() for path, t in tensors.items()},
                    'cpu': {path: (kind, data) for path, (kind, _, data) in cpu.items()},
                    'kv': {}, 'cpu_rng': torch.get_rng_state(),
                    'xpu_rng': torch.xpu.get_rng_state(), 'random_rng': random.getstate(),
                    'numpy_rng': np.random.get_state()}
        for path, (tensor, dim, indices) in kv.items():
            index = torch.tensor(indices, device=tensor.device, dtype=torch.int64)
            selected = tensor.view(torch.uint8) if tensor.dtype == torch.float8_e4m3fn else tensor
            snapshot['kv'][path] = {'dim': dim, 'indices': indices, 'dtype': tensor.dtype,
                'data': selected.index_select(dim, index).cpu().clone()}
        return snapshot, refs

    def restore(runner, snapshot, refs=None):
        if refs:
            for setter, value in refs.values():
                setter(value)
        tensors, cpu, _, kv = registry(runner)
        differences = {kind: {'missing': [repr(x) for x in set(saved) - set(local)],
                              'extra': [repr(x) for x in set(local) - set(saved)]}
                       for kind, local, saved in (('tensor', tensors, snapshot['tensors']),
                                                 ('cpu', cpu, snapshot['cpu']), ('kv', kv, snapshot['kv']))
                       if set(local) != set(saved)}
        if differences:
            raise RuntimeError('mutable state registry differs: ' + repr(differences))
        for path, (kind, value) in snapshot['cpu'].items():
            local_kind, local, _ = cpu[path]
            if kind != local_kind:
                raise RuntimeError('CPU state kind differs: ' + repr(path))
            if kind == 'array':
                if local.shape != value.shape:
                    raise RuntimeError('CPU state shape differs: ' + repr(path))
                local[...] = value
            else:
                local(copy.deepcopy(value))
        for path, value in snapshot['tensors'].items():
            target = tensors[path]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise RuntimeError('buffer shape/dtype differs: ' + repr(path))
            target.copy_(value)
        for path, saved in snapshot['kv'].items():
            tensor, dim, _ = kv[path]
            if dim != saved['dim'] or tensor.dtype != saved['dtype']:
                raise RuntimeError('KV layout differs: ' + repr(path))
            index = torch.tensor(saved['indices'], device=tensor.device, dtype=torch.int64)
            target = tensor.view(torch.uint8) if tensor.dtype == torch.float8_e4m3fn else tensor
            target.index_copy_(dim, index, saved['data'].to(tensor.device))
        torch.set_rng_state(snapshot['cpu_rng'])
        torch.xpu.set_rng_state(snapshot['xpu_rng'])
        random.setstate(snapshot['random_rng'])
        np.random.set_state(snapshot['numpy_rng'])
        torch.xpu.synchronize()

    def verify_restored(runner, snapshot):
        tensors, cpu, _, kv = registry(runner)
        for path, expected in snapshot['tensors'].items():
            actual = tensors[path].detach().cpu().contiguous()
            if not torch.equal(actual.reshape(-1).view(torch.uint8), expected.contiguous().reshape(-1).view(torch.uint8)):
                raise RuntimeError('restored buffer bytes differ: ' + repr(path))
        for path, expected in snapshot['kv'].items():
            tensor, dim, _ = kv[path]
            index = torch.tensor(expected['indices'], device=tensor.device, dtype=torch.int64)
            selected = tensor.view(torch.uint8) if tensor.dtype == torch.float8_e4m3fn else tensor
            actual = selected.index_select(dim, index).cpu().contiguous()
            if not torch.equal(actual.reshape(-1).view(torch.uint8), expected['data'].contiguous().reshape(-1).view(torch.uint8)):
                raise RuntimeError('restored KV/GDN bytes differ: ' + repr(path))
        for path, (kind, expected) in snapshot['cpu'].items():
            _, _, actual = cpu[path]
            if kind == 'array':
                equal = actual.dtype == expected.dtype and actual.shape == expected.shape and actual.tobytes() == expected.tobytes()
            else:
                equal = actual == expected
            if not equal:
                raise RuntimeError('restored CPU bookkeeping differs: ' + repr(path))

    original_prepare = GPUModelRunner.prepare_attn

    def prepare(self, input_batch):
        if state['replaying']:
            state['prepared_inputs'] = {key: getattr(input_batch, key).clone()
                for key in ('input_ids', 'positions', 'seq_lens', 'query_start_loc', 'idx_mapping', 'is_padding')}
        return original_prepare(self, input_batch)

    original_execute = GPUModelRunner.execute_model
    original_sample_tokens = GPUModelRunner.sample_tokens
    original_sample = GPUModelRunner.sample
    original_propose = AutoRegressiveSpeculator.propose

    def sample(self, *args, **kwargs):
        result = original_sample(self, *args, **kwargs)
        if state['pending'] is not None:
            output, sampled, rejected = result
            saved = state['canonical']
            if saved is None:
                state['pending']['commit'] = (output.sampled_token_ids.cpu().clone(),
                                             sampled.cpu().clone(), rejected.cpu().clone())
            else:
                if state['replaying']:
                    for target, source in zip(saved['observed_commit_device'], (output.sampled_token_ids, sampled, rejected)):
                        target.copy_(source)
                for target, source in zip((output.sampled_token_ids, sampled, rejected), saved.get('commit_device', saved['commit'])):
                    target.copy_(source)
        return result

    def propose(self, *args, **kwargs):
        if state['pending'] is not None:
            # Four positional tensor inputs determine accepted positions and the
            # actual target hidden state delivered to the first MTP forward.
            saved = state['canonical']
            if saved is None:
                state['pending']['draft_inputs'] = {i: args[i].cpu().clone() for i in (3, 5, 6, 7, 8)}
            else:
                for i, source in saved.get('draft_inputs_device', saved['draft_inputs']).items():
                    args[i].copy_(source)
        return original_propose(self, *args, **kwargs)

    def execute(self, scheduler_output, *args, **kwargs):
        if state['replaying'] or state['pending'] is not None or not (root / 'cycle-replay-active').exists():
            return original_execute(self, scheduler_output, *args, **kwargs)
        count = len(scheduler_output.num_scheduled_tokens)
        query = list(scheduler_output.num_scheduled_tokens.values())
        if count not in (1, 4) or count in state['seen'] or query != [5] * count or scheduler_output.scheduled_new_reqs:
            return original_execute(self, scheduler_output, *args, **kwargs)
        torch.xpu.synchronize()
        pre, refs = capture(self)
        pending = {'count': count, 'pre': pre, 'refs': refs,
                   'scheduler': copy.deepcopy(scheduler_output), 'args': args, 'kwargs': kwargs}
        fixture = fixtures / f'c{count}.pt'
        state['canonical'] = torch.load(fixture, weights_only=False, map_location='cpu', mmap=True) if fixture.exists() else None
        if state['canonical'] is not None:
            # Scratch staging pool capacities depend on earlier prefills. Their
            # live values are regenerated; active .gpu views and source .cpu/.np
            # arrays are restored, along with each pool's ring index.
            for section in ('tensors', 'cpu'):
                saved = state['canonical']['pre'][section]
                state['canonical']['pre'][section] = {path: value for path, value in saved.items() if '_uva_bufs' not in path}
            pending.pop('pre')  # the local pre-snapshot is unnecessary once a canonical fixture exists
        state['pending'] = pending
        return original_execute(self, scheduler_output, *args, **kwargs)

    def sample_tokens(self, grammar_output):
        if state['pending'] is not None and not state['replaying'] and state['canonical'] is None:
            state['pending']['target_hidden'] = self.execute_model_state.hidden_states.cpu().clone()
        output = original_sample_tokens(self, grammar_output)
        pending = state['pending']
        if pending is None or state['replaying']:
            return output
        torch.xpu.synchronize()
        extra = {path: value['indices'] for path, value in state['canonical']['pre']['kv'].items()} if state['canonical'] else None
        post, post_refs = capture(self, extra)
        fixture = fixtures / f"c{pending['count']}.pt"
        if state['canonical'] is None:
            canonical = {key: pending[key] for key in ('pre', 'scheduler', 'commit', 'draft_inputs', 'target_hidden')}
            torch.save(canonical, fixture)
            state['canonical'] = canonical
        canonical = state['canonical']
        canonical['commit_device'] = tuple(x.to(self.device) for x in canonical['commit'])
        canonical['observed_commit_device'] = tuple(torch.empty_like(x) for x in canonical['commit_device'])
        canonical['head_hidden_device'] = canonical['target_hidden'].to(self.device)
        canonical['draft_inputs_device'] = {i: x.to(self.device) for i, x in canonical['draft_inputs'].items()}
        rows, signatures = [], []
        state['replaying'] = True
        try:
            for repetition in range(15):
                # Each iteration starts from the identical canonical semantic
                # state. Process-specific tensor addresses remain local.
                restore(self, canonical['pre'], pending['refs'])
                self.execute_model_state = None
                verify_restored(self, canonical['pre'])
                start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
                schedule = copy.deepcopy(canonical['scheduler'])
                wall = time.perf_counter()
                start.record()
                original_execute(self, schedule, *pending['args'], **pending['kwargs'])
                hidden = self.execute_model_state.hidden_states.clone()
                # Execute the real target first and observe its output. Then
                # hold the head and sampler input fixed as well as the later
                # draft input. All these identical copies remain in the timer.
                self.execute_model_state.hidden_states.copy_(canonical['head_hidden_device'])
                result = original_sample_tokens(self, grammar_output)
                end.record()
                end.synchronize()
                elapsed = 1000 * (time.perf_counter() - wall)
                # Observe after timing, never force output in lieu of executing.
                target_sig = hashlib.sha256(hidden[:5 * pending['count']].cpu().numpy().tobytes()).hexdigest()
                sig = hashlib.sha256(self.req_states.draft_tokens.cpu().numpy().tobytes()).hexdigest()
                signatures.append(sig)
                prepared = {key: tensor.cpu().tolist() for key, tensor in state['prepared_inputs'].items()}
                input_signature = hashlib.sha256(json.dumps(prepared, sort_keys=True).encode()).hexdigest()
                observed = tuple(x.cpu() for x in canonical['observed_commit_device'])
                expected = canonical['commit']
                active = torch.arange(expected[0].shape[1])[None, :] < expected[1][:, None]
                commit_matches = (torch.equal(observed[1], expected[1]) and torch.equal(observed[2], expected[2])
                                  and torch.equal(observed[0][active], expected[0][active]))
                rows.append({'repeat': repetition, 'wall_ms': elapsed,
                             'stream_ms': start.elapsed_time(end), 'draft_signature': sig, 'target_hidden_signature': target_sig,
                             'target_bitwise_equal_reference': torch.equal(hidden.cpu(), canonical['target_hidden']),
                             'input_signature': input_signature, 'prepared_inputs': prepared,
                             'sampler_commit_matches_reference': commit_matches,
                             'draft_tokens': self.req_states.draft_tokens.cpu().tolist(),
                             'target_max_abs_reference': float((hidden.cpu().float() - canonical['target_hidden'].float()).abs().max())})
                del result
            valid = (len(set(signatures)) == 1 and len({row['input_signature'] for row in rows}) == 1
                     and all(row['sampler_commit_matches_reference'] for row in rows))
            (root / f"cycle-replay-c{pending['count']}.json").write_text(json.dumps({
                'scope': 'execute_model + sample_tokens including target, sampler, commit and MTP4 draft',
                'repeatability_passed': valid,
                'registered_initial_state_bytes_verified_each_repeat': True,
                'head_and_sampler_inputs_frozen': True,
                'real_sampler_commit_matches_reference_each_repeat': all(row['sampler_commit_matches_reference'] for row in rows),
                'target_bitwise_repeatable': len({row['target_hidden_signature'] for row in rows}) == 1,
                'target_numeric_note': 'Target output is observed before head/sampler/commit/draft input freezing. Raw target variation is retained; initial state bytes and real sampler commits must match.',
                'request_count': pending['count'], 'actual_query_rows': 5 * pending['count'],
                'fixture_sha256': hashlib.file_digest(fixture.open('rb'), 'sha256').hexdigest(),
                'fixed_commit': [x.tolist() for x in canonical['commit']],
                'restore_outside_timing': True, 'rows': rows,
                'mutable_tensor_count': len(canonical['pre']['tensors']),
                'mutable_kv_views': len(canonical['pre']['kv']),
                'borrowed': 'immutable model weights, local graph handles/pointer tables, staging-pool scratch overwritten by each step; active views, source arrays and pool indices are restored',
                'sampled_commit_and_draft_input_copies_in_timing': True}, indent=2))
            if not valid:
                raise RuntimeError('restored-cycle input/target/draft is not repeatable; diagnostic rows saved')
        finally:
            restore(self, post, post_refs)
            self.execute_model_state = None
            state['seen'].add(pending['count'])
            state.update(replaying=False, pending=None, canonical=None)
        return output

    GPUModelRunner.prepare_attn = prepare
    GPUModelRunner.execute_model = execute
    GPUModelRunner.sample_tokens = sample_tokens
    GPUModelRunner.sample = sample
    AutoRegressiveSpeculator.propose = propose
    globals()['snapshot_capture'] = capture
    globals()['snapshot_restore'] = restore
    (root / f'cycle-installed-{os.getpid()}').touch()


if os.environ.get('B70_CYCLE_REPLAY') == '1' and not any('cpuinfo' in arg for arg in sys.argv):
    install()

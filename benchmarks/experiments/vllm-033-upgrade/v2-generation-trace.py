"""Diagnostic-only V2 trace; device summaries read at existing output copy event.

No RNG changes, no new explicit synchronization, no model/request overrides.
Extra diagnostic kernels/copies can still perturb timing. Never a release patch.
"""
import inspect
import hashlib
import json
import os
import re
from pathlib import Path
import torch
from vllm.triton_utils import triton, tl

_pending = []
_current = None
_calls = dict(v2_model_sample=0, v2_rejection=0, legacy_v1_rejection=0)

@triton.jit
def _stats(X, OUT, S0: tl.constexpr, S1: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, B)
    nan = tl.full((), 0, tl.int32)
    posinf = tl.full((), 0, tl.int32)
    neginf = tl.full((), 0, tl.int32)
    finite = tl.full((), 0, tl.int32)
    maxval = tl.full((), -float('inf'), tl.float32)
    minval = tl.full((), float('inf'), tl.float32)
    for start in range(tl.cdiv(N, B)):
        col = start * B + lane
        valid = col < N
        x = tl.load(X + row * S0 + col * S1, valid, other=0).to(tl.float32)
        good = valid & (x == x) & (x != float('inf')) & (x != -float('inf'))
        nan += tl.sum((valid & (x != x)).to(tl.int32))
        posinf += tl.sum((valid & (x == float('inf'))).to(tl.int32))
        neginf += tl.sum((valid & (x == -float('inf'))).to(tl.int32))
        finite += tl.sum(good.to(tl.int32))
        maxval = tl.maximum(maxval, tl.max(tl.where(good, x, -float('inf')), 0))
        minval = tl.minimum(minval, tl.min(tl.where(good, x, float('inf')), 0))
    tl.store(OUT + row * 6 + 0, nan)
    tl.store(OUT + row * 6 + 1, posinf)
    tl.store(OUT + row * 6 + 2, neginf)
    tl.store(OUT + row * 6 + 3, finite)
    tl.store(OUT + row * 6 + 4, maxval)
    tl.store(OUT + row * 6 + 5, minval)

def tensor(name, value):
    if _current is None or value is None or value.numel() == 0:
        return
    if value.ndim != 2:
        raise ValueError((name, value.shape))
    out = torch.empty((value.shape[0], 6), dtype=torch.float32, device=value.device)
    _stats[(value.shape[0],)](value, out, value.stride(0), value.stride(1), value.shape[1], B=4096)
    _current['tensors'][name] = out
    _current['metadata'][name] = dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype))

def small(name, value):
    if _current is not None and value is not None:
        _current['tensors'][name] = value.clone()

def identity(obj):
    cls = type(obj); source = inspect.getsourcefile(cls)
    return dict(klass=cls.__qualname__, module=cls.__module__, source=source,
                sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None,
                signature=(str(inspect.signature(cls.__call__)) if hasattr(cls, '__call__') else None))

def begin(runner, hidden, logits, batch):
    global _current
    _calls['v2_model_sample'] += 1
    if _calls['v2_model_sample'] > 3500:
        _current = None
        return
    prior = _current if _current is not None and _current['metadata'].get('layer_trace') else None
    _current = dict(metadata=dict(step=_calls['v2_model_sample'], req_ids=list(batch.req_ids),
        num_reqs=batch.num_reqs, num_draft_tokens=batch.num_draft_tokens,
        idx_mapping_np=batch.idx_mapping_np.tolist(), cu_num_logits_np=batch.cu_num_logits_np.tolist(),
        num_computed_tokens_np=batch.num_computed_tokens_np.tolist(),
        prefill_len_np=batch.prefill_len_np.tolist(), query_start_loc_np=batch.query_start_loc_np.tolist(),
        num_scheduled_tokens=batch.num_scheduled_tokens.tolist(),
        num_draft_tokens_per_req=(None if batch.num_draft_tokens_per_req is None else batch.num_draft_tokens_per_req.tolist()),
        calls=dict(_calls)), tensors={})
    if prior is not None:
        _current['metadata'].update(prior['metadata'])
        _current['tensors'].update(prior['tensors'])
    if _calls['v2_model_sample'] == 1:
        _current['metadata']['runner'] = identity(runner)
        _current['metadata']['sampler'] = identity(runner.sampler)
        _current['metadata']['rejection_sampler'] = identity(runner.rejection_sampler) if runner.rejection_sampler else None
        _current['metadata']['model_state'] = identity(runner.model_state)
    identity_path = Path('/results/v2-trace/identity.json')
    if not identity_path.exists():
        identity_path.parent.mkdir(exist_ok=True)
        identity_path.write_text(json.dumps(dict(runner=identity(runner), sampler=identity(runner.sampler), rejection_sampler=identity(runner.rejection_sampler) if runner.rejection_sampler else None, model_state=identity(runner.model_state)), indent=2)+'\n')
    for name in ('logits_indices', 'positions', 'input_ids', 'cu_num_logits', 'idx_mapping', 'expanded_idx_mapping', 'expanded_local_pos', 'seq_lens'):
        small(name, getattr(batch, name))
    small('state_num_computed_tokens', runner.req_states.num_computed_tokens.gpu)
    tensor('hidden', hidden)
    tensor('target_raw', logits)

def reject(sampler, batch, draft_logits):
    _calls['v2_rejection'] += 1
    if _current is None:
        return
    _current['metadata']['calls'] = dict(_calls)
    _current['metadata']['draft_logits_none'] = draft_logits is None
    if draft_logits is not None:
        counts = batch.num_draft_tokens_per_req
        for i, state_idx in enumerate(batch.idx_mapping_np):
            count = int(counts[i]) if counts is not None else 0
            # Only the actually scheduled draft slots; -inf in pruned vocabulary is intentional.
            if count:
                tensor('draft_state_'+str(state_idx), draft_logits[int(state_idx), :count])

def legacy_call():
    _calls['legacy_v1_rejection'] += 1

def finish(output):
    if _current is None:
        return
    small('sampled_token_ids', output.sampled_token_ids)
    small('num_sampled', output.num_sampled)
    small('num_rejected', output.num_rejected)
    _pending.append(_current)

def enqueue(output):
    global _pending, _current
    output._b70_trace = _pending
    _pending = []
    _current = None
    # Called within the existing copy-stream context after its wait_stream(main).
    for record in output._b70_trace:
        record['cpu'] = {}
        for name, value in record['tensors'].items():
            cpu = torch.empty(value.shape, dtype=value.dtype, device='cpu', pin_memory=True)
            cpu.copy_(value, non_blocking=True)
            record['cpu'][name] = cpu

def flush(output):
    # Called immediately AFTER the pre-existing self.copy_event.synchronize().
    records = getattr(output, '_b70_trace', [])
    if not records:
        return
    folder = Path('/results/v2-trace'); folder.mkdir(exist_ok=True)
    with (folder / ('trace-'+str(os.getpid())+'.jsonl')).open('a') as stream:
        for record in records:
            row = dict(record['metadata'])
            row['values'] = {name:value.tolist() for name,value in record['cpu'].items()}
            stream.write(json.dumps(row, allow_nan=True)+'\n')
    output._b70_trace = []


_layer_active = False
_hooked = False

def _hook_value(name, value):
    if not _layer_active:
        return
    if isinstance(value, torch.Tensor) and value.ndim == 2:
        tensor(name, value)
    elif isinstance(value, (tuple, list)):
        for i, x in enumerate(value):
            _hook_value(name+'_'+str(i), x)

def before_forward(runner, batch, attn_metadata, block_tables, dummy):
    global _layer_active, _hooked, _current
    _layer_active = False
    if dummy:
        return
    if not _hooked:
        names = []
        for name, module in runner.model.named_modules():
            if re.search(r'(?:^|\.)layers\.\d+(?:\.(?:linear_attn|self_attn|mlp|input_layernorm|post_attention_layernorm))?$', name):
                def pre(mod, args, kwargs, key=name):
                    _hook_value('layer_'+key+'_input', args)
                    for k,v in kwargs.items():
                        _hook_value('layer_'+key+'_input_'+k, v)
                def post(mod, args, kwargs, output, key=name):
                    _hook_value('layer_'+key+'_output', output)
                module.register_forward_pre_hook(pre, with_kwargs=True)
                module.register_forward_hook(post, with_kwargs=True)
                names.append(name)
        folder=Path('/results/v2-trace');folder.mkdir(exist_ok=True)
        (folder/'layer-hook-inventory.json').write_text(json.dumps(names,indent=2)+'\n')
        _hooked=True
    # CPU mirror only chooses diagnostic window; actual positions are also captured.
    _layer_active = any(2280 <= int(x) <= 2310 for x in batch.num_computed_tokens_np)
    if not _layer_active:
        return
    _current = dict(metadata=dict(layer_trace=True), tensors={})
    modelstate=runner.model_state
    for name in ('num_accepted_tokens_gpu','_mamba_state_idx_gpu','_mamba_src_col_gpu','_mamba_src_off_gpu'):
        value=getattr(modelstate,name,None)
        if isinstance(value,torch.Tensor):small('mamba_'+name,value)
    for i, table in enumerate(block_tables or ()):
        small('block_table_'+str(i), table[:batch.num_reqs])
    if isinstance(attn_metadata,dict):
        for name,meta in attn_metadata.items():
            fields={}
            for key in ('block_table_tensor','query_start_loc','seq_lens','num_accepted_tokens','has_initial_state','state_indices_tensor','spec_state_indices_tensor','non_spec_state_indices_tensor'):
                value=getattr(meta,key,None)
                if isinstance(value,torch.Tensor):small('attn_'+name+'_'+key,value)
                elif isinstance(value,(int,float,str,bool)):fields[key]=value
            _current['metadata']['attn_'+name]=fields

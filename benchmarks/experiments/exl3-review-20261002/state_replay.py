"""Bounded, destructive-to-timing diagnostic on the actual V2 target step.

Each shadow route starts with the identical active KV and GDN cache blocks,
inputs, metadata and batch padding. The actual serving call runs last, after
restoring all saved state. This is never installed in a production image.
"""
import ast
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap
import types

import numpy as np
import torch

RUNNER_SHA256 = '174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce'


def tensor_hash(t):
    # Singleton views may be "contiguous" while retaining a non-unit stride.
    # An explicit flat destination normalizes this before the byte reinterpret.
    flat = torch.empty(t.numel(),dtype=t.dtype,device='cpu')
    flat.copy_(t.detach().cpu().reshape(-1))
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()


def flatten_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for part in value:
            yield from flatten_tensors(part)
    else:
        raise TypeError(f'Unknown cache representation: {type(value)}')


def storage_hash(value):
    digest = hashlib.sha256()
    for begin in range(0,value.numel(),16*1024*1024):
        cpu=value[begin:begin+16*1024*1024].cpu()
        digest.update(memoryview(cpu.numpy()))
    return digest.hexdigest()


class State:
    def __init__(self, runner, md, slots, inputs, full=False):
        torch.xpu.synchronize()
        layers = runner.vllm_config.compilation_config.static_forward_context
        assert isinstance(md, dict) and isinstance(slots, dict)
        self.saved = []
        self.inputs = []
        self.full_saved = []
        self.inventory = []
        entries = {}
        for name, metadata in md.items():
            assert metadata is not None
            layer = layers[name]
            cache = list(flatten_tensors(layer.kv_cache))
            indices = set()
            if hasattr(metadata, 'block_table'):
                table = metadata.block_table.cpu()
                lengths = metadata.seq_lens.cpu()
                group = next(g for g in runner.kv_cache_config.kv_cache_groups if name in g.layer_names)
                block_size = group.kv_cache_spec.block_size
                for row, length in zip(table, lengths):
                    indices.update(int(x) for x in row[:(int(length)+block_size-1)//block_size])
                write_slots = slots[name].cpu().flatten()
                indices.update(int(x)//block_size for x in write_slots if int(x) >= 0)
                kind = 'attention'
            else:
                for attr in ('spec_state_indices_tensor', 'non_spec_state_indices_tensor', 'prefill_state_indices'):
                    value = getattr(metadata, attr, None)
                    if isinstance(value, torch.Tensor):
                        indices.update(int(x) for x in value.cpu().flatten() if int(x) >= 0)
                kind = 'GDN'
            assert indices, f'No active state slots for {name}'
            indices = sorted(indices)
            for part, value in enumerate(cache):
                assert value.device.type == 'xpu' and max(indices) < value.shape[0], (name, value.shape, indices)
                key = (value.data_ptr(), tuple(value.shape), tuple(value.stride()))
                entry = entries.setdefault(key,dict(value=value,indices=set(),owners=[]))
                entry['indices'].update(indices)
                entry['owners'].append(dict(layer=name,kind=kind,part=part))
        # Shared target/draft state can appear under several layer names.
        # Snapshot/restore the union once; repeated partial restores are unsafe.
        for entry in entries.values():
            value = entry['value']
            indices = sorted(entry['indices'])
            idx = torch.tensor(indices,dtype=torch.int64,device=value.device)
            saved = value.index_select(0,idx).cpu()
            if not full:
                self.saved.append((value,idx,saved))
            self.inventory.append(dict(layer=entry['owners'][0]['layer'],kind=entry['owners'][0]['kind'],
                owners=entry['owners'],shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype),
                active_indices=indices,saved_bytes=saved.numel()*saved.element_size(),before_sha256=tensor_hash(saved)))
        assert sum(x['saved_bytes'] for x in self.inventory) < 4_000_000_000
        if full:
            storages={entry['value'].untyped_storage().data_ptr():entry['value'].untyped_storage() for entry in entries.values()}
            assert sum(s.nbytes() for s in storages.values()) < 14_000_000_000
            for storage in storages.values():
                view=torch.empty(0,dtype=torch.uint8,device='xpu').set_(storage,0,(storage.nbytes(),),(1,))
                saved=view.cpu()
                self.full_saved.append((view,saved,storage_hash(saved)))
        # Fused residual operations can write into the embeddings supplied by
        # the multimodal runner. Same pointers alone do not imply same inputs.
        seen_inputs = set()
        values = [('model-input/'+key,value) for key,value in inputs.items()]
        values += [('metadata/'+name+'/'+key,value) for name,obj in md.items() for key,value in vars(obj).items()]
        values += [('slot-map/'+key,value) for key,value in slots.items()]
        for name,value in values:
            if not isinstance(value,torch.Tensor):
                continue
            key = (value.data_ptr(),tuple(value.shape),tuple(value.stride()))
            if key in seen_inputs:
                continue
            seen_inputs.add(key)
            saved = value.detach().cpu().clone()
            digest = tensor_hash(saved)
            self.inputs.append((name,value,saved,digest))
        self.initial_digest = hashlib.sha256(json.dumps(self.inventory, sort_keys=True).encode()).hexdigest()
        assert {'attention', 'GDN'} == {x['kind'] for x in self.inventory}

    def restore(self):
        for value,saved,digest in self.full_saved:
            value.copy_(saved)
        for value, idx, saved in self.saved:
            value.index_copy_(0, idx, saved.to(value.device))
        for name,value,saved,digest in self.inputs:
            value.copy_(saved)
        torch.xpu.synchronize()

    def verify(self):
        for value,saved,digest in self.full_saved:
            assert storage_hash(value)==digest, 'Full underlying cache storage was not restored'
        for (value, idx, saved), entry in zip(self.saved, self.inventory):
            assert tensor_hash(value.index_select(0, idx)) == entry['before_sha256'], entry['layer']
        for name,value,saved,digest in self.inputs:
            assert tensor_hash(value) == digest, name
        return self.initial_digest


def compare_logits(a, b):
    assert a.shape == b.shape and a.shape[-1] == 248320
    af, bf = a.float(), b.float()
    assert torch.isfinite(af).all() and torch.isfinite(bf).all()
    d = bf-af
    ap, bp = af.log_softmax(-1), bf.log_softmax(-1)
    top = af.topk(2, dim=-1)
    bt = bf.topk(2, dim=-1)
    return dict(max_abs=float(d.abs().max()), rms=float(d.square().mean().sqrt()),
        kl_native_to_route_mean=float((ap.exp()*(ap-bp)).sum(-1).mean()),
        rows=[dict(row=i, native_top1=int(top.indices[i,0]), route_top1=int(bt.indices[i,0]),
            native_margin=float(top.values[i,0]-top.values[i,1]), route_margin=float(bt.values[i,0]-bt.values[i,1]),
            delta_at_native_winner=float(d[i,top.indices[i,0]]), delta_at_route_winner=float(d[i,bt.indices[i,0]]))
            for i in range(a.shape[0])])


class Replay:
    def __init__(self, runner, output):
        self.runner = runner
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.steps = []
        self.enabled = False
        self.case = None
        self.wanted = set()

    def arm(self, case):
        self.case = case
        self.enabled = True
        self.wanted = {'prefill', 'early-mixed', 'c4-verification'} if case == 'prose-4096' else {'long-prefill'}
        if os.environ.get('B70_REPLAY_FULL_MIXED_STATE')=='1':
            assert case=='prose-4096'
            self.wanted={'early-mixed'}
        if os.environ.get('B70_REPLAY_EARLY_TARGETS')=='1':
            assert case=='prose-4096'
            self.wanted={'early-request2-index1','early-request3-index3'}

    def select(self, batch):
        self.selected_targets=[]
        lengths = batch.num_scheduled_tokens.tolist()
        computed = batch.num_computed_prefill_tokens_np.tolist()
        prefill = batch.prefill_len_np.tolist()
        if os.environ.get('B70_REPLAY_EARLY_TARGETS')=='1':
            positions=batch.positions.cpu()
            if positions.ndim==2:positions=positions[0]
            offset=0
            for req_id,length,prefill_length in zip(batch.req_ids,lengths,prefill):
                index=int(req_id.split('-',1)[0])
                for request,output_index in [(2,1),(3,3)]:
                    label=f'early-request{request}-index{output_index}'
                    position=prefill_length+output_index-1
                    if index==request and label in self.wanted and position in positions[offset:offset+length].tolist():
                        self.selected_targets.append(dict(label=label,request_index=request,output_index=output_index,
                            absolute_position=int(position),target_input_row=offset+positions[offset:offset+length].tolist().index(position)))
                offset+=length
            if self.selected_targets:return '+'.join(t['label'] for t in self.selected_targets)
            return None
        if 'prefill' in self.wanted and max(lengths) >= 64 and max(c+q for c,q in zip(computed,lengths)) >= 4096:
            return 'prefill'
        if 'long-prefill' in self.wanted and max(lengths) >= 64 and max(computed) >= 4096:
            return 'long-prefill'
        if 'early-mixed' in self.wanted and batch.has_prefill and any(c >= p and c-p <= 4 for c,p in zip(computed,prefill)):
            return 'early-mixed'
        if 'c4-verification' in self.wanted and not batch.has_prefill and batch.num_reqs == 4 and lengths == [4]*4:
            return 'c4-verification'
        return None

    def call(self, batch, desc, inputs, md, slots, function, args, kwargs):
        label = self.select(batch) if self.enabled else None
        if label is None:
            return function(*args, **kwargs)
        if self.selected_targets:
            for target in self.selected_targets:self.wanted.remove(target['label'])
        else:self.wanted.remove(label)
        runner = self.runner
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context, BatchDescriptor
        from vllm.v1.attention.backends import flash_attn as fa
        from exl3xpu import attention_dispatch as dispatch, shared_kv_verify as verify
        guarded = fa.flash_attn_varlen_func
        cells = dict(zip(guarded.__code__.co_freevars, guarded.__closure__))
        assert {'original', 'context_var', 'library', 'onednn', 'native_op'} <= set(cells)
        saved_cells = {key: cells[key].cell_contents for key in ('library', 'onednn')}
        original = cells['original'].cell_contents
        full=(os.environ.get('B70_REPLAY_FULL_MIXED_STATE')=='1' or os.environ.get('B70_REPLAY_EARLY_TARGETS')=='1')
        state = State(runner, md, slots, inputs,full=full)
        item = dict(case=self.case, label=label, req_ids=list(batch.req_ids),
            actual_rows=int(batch.num_tokens), padded_rows=int(batch.num_tokens_after_padding),
            rows_by_request=batch.num_scheduled_tokens.tolist(), computed=batch.num_computed_prefill_tokens_np.tolist(),
            prefill_lengths=batch.prefill_len_np.tolist(), graph_mode=str(desc.cg_mode),
            positions=inputs['positions'].cpu().tolist(), input_ids=batch.input_ids.cpu().tolist(),
            model_input_tensors={key:dict(shape=list(value.shape),dtype=str(value.dtype),sha256=tensor_hash(value))
                                 for key,value in inputs.items() if isinstance(value,torch.Tensor)},
            logits_indices=batch.logits_indices.cpu().tolist(), state=state.inventory,
            restored_state_sha256=state.initial_digest, routes={})
        item['full_storage_snapshots']=[dict(bytes=v.numel(),sha256=digest) for v,s,digest in state.full_saved]
        item['selected_early_targets']=list(self.selected_targets)
        prefix = f'{len(self.steps):02d}-{self.case}-{label}'
        (self.output/(prefix+'-before.json')).write_text(json.dumps(item, indent=2)+'\n')
        logits = {}
        gdn_captures = {}
        original_gdn = torch.ops._xpu_C.gdn_attention
        try:
            for route, use_m04, use_onednn in [('native',False,False),('native-repeat',False,False),('m04',True,False),('onednn',False,True),('combined',True,True)]:
                state.restore()
                assert state.verify() == state.initial_digest
                cells['library'].cell_contents = saved_cells['library'] if use_m04 else None
                cells['onednn'].cell_contents = use_onednn
                local_checks = []
                routes = Counter()
                capturing = desc.cg_mode == CUDAGraphMode.FULL
                gdn_calls = [0]
                captured_gdn = {}
                if full and os.environ.get('B70_REPLAY_LOCATE_NATIVE_MIXED')=='1' and route.startswith('native'):
                    assert not capturing
                    def observe_gdn(core,z,qkv,ba,*a,**kw):
                        first=gdn_calls[0]==0
                        if first:
                            captured_gdn.update(projected_qkv=qkv.detach().clone(),projected_ba=ba.detach().clone())
                        result=original_gdn(core,z,qkv,ba,*a,**kw)
                        if first:
                            captured_gdn.update(core_output=core.detach().clone(),output_gate=z.detach().clone())
                        gdn_calls[0]+=1
                        return result
                    torch.ops._xpu_C.gdn_attention=observe_gdn

                def observed(*positional, **d):
                    context = cells['context_var'].cell_contents.get()
                    if not positional and context.allowed:
                        chosen = 'native'
                        if use_onednn and dispatch.prefill_eligible(d,context,torch.xpu.is_current_stream_capturing()):
                            chosen = 'mixed/oneDNN'
                        elif use_m04 and verify.eligible(d) and d['cu_seqlens_q'].numel()-1 <= 4:
                            chosen = 'M04'
                        routes[chosen] += 1
                    else:
                        chosen = 'native'
                        routes[chosen] += 1
                    result = original(*positional, **d) if route.startswith('native') else guarded(*positional, **d)
                    if chosen != 'native':
                        reference = torch.empty_like(d['out'])
                        original(**dict(d,out=reference))
                        diff = (d['out'].float()-reference.float()).abs()
                        bad = diff > (.003+.01*reference.float().abs())
                        local_checks.append(torch.stack((diff.max(),bad.sum().float(),(~torch.isfinite(d['out'])).sum().float())))
                    return result

                fa.flash_attn_varlen_func = observed

                def body():
                    with set_forward_context(md,runner.vllm_config,
                            num_tokens=batch.num_tokens_after_padding,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE,
                            batch_descriptor=BatchDescriptor(num_tokens=batch.num_tokens_after_padding),
                            slot_mapping=slots,is_padding=batch.is_padding):
                        return runner.model(**inputs)

                if capturing:
                    # Warm/capture always from the saved state. These graphs are
                    # diagnostic copies of this exact real verification batch.
                    for _ in range(2):
                        body(); torch.xpu.synchronize(); state.restore()
                    local_checks.clear(); routes.clear()
                    graph = torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph):
                        hidden = body()
                    state.restore()
                    assert state.verify() == state.initial_digest
                    graph.replay()
                else:
                    hidden = body()
                hidden = hidden[0] if isinstance(hidden,tuple) else hidden
                target = runner.model.compute_logits(hidden[batch.logits_indices])
                value = target.detach().cpu()
                torch.ops._xpu_C.gdn_attention=original_gdn
                if captured_gdn:
                    gdn_captures[route]={key:value.cpu() for key,value in captured_gdn.items()}
                    item['routes'].setdefault(route,{})
                    item.setdefault('first_gdn_native_hashes',{})[route]={key:tensor_hash(value) for key,value in gdn_captures[route].items()}
                    captured_gdn.clear()
                assert torch.isfinite(value).all()
                logits[route] = value
                np.save(self.output/(prefix+'-'+route+'-logits.npy'),value.numpy())
                check = torch.stack(local_checks).cpu().tolist() if local_checks else []
                item['routes'][route] = dict(shape=list(value.shape),sha256=tensor_hash(value),
                    mutated_model_inputs=[name for name,t,s,digest in state.inputs if name.startswith('model-input/') and tensor_hash(t)!=digest],
                    actual_routes=dict(routes), local_attention_checks=check,
                    local_original_tolerance_pass=all(x[1] == 0 and x[2] == 0 for x in check),
                    execution='new XPU graph replay of actual verification batch, with native shadow attention' if capturing else 'eager actual target batch, with native shadow attention',
                    initial_state_verified_sha256=state.initial_digest)
                del target,hidden
                if capturing:
                    del graph
                print('MATCHED_STATE_REPLAY',self.case,label,route,item['routes'][route],flush=True)
            item['comparisons'] = {route:compare_logits(logits['native'],value) for route,value in logits.items() if route != 'native'}
            if gdn_captures:
                assert set(gdn_captures)=={'native','native-repeat'}
                differences={}
                for key,a in gdn_captures['native'].items():
                    b=gdn_captures['native-repeat'][key];diff=b.float()-a.float()
                    per_request=[];start=0
                    for count in item['rows_by_request']:
                        part=diff[start:start+count];per_request.append(dict(max_abs=float(part.abs().max()),rms=float(part.square().mean().sqrt())))
                        start+=count
                    differences[key]=dict(bitidentical=tensor_hash(a)==tensor_hash(b),shape=list(a.shape),
                        max_abs=float(diff.abs().max()),rms=float(diff.square().mean().sqrt()),per_request=per_request)
                item['first_gdn_native_repeat_differences']=differences
            item['status'] = 'PASS_LOCAL_ATTENTION_ORIGINAL_TOLERANCE' if all(r['local_original_tolerance_pass'] for r in item['routes'].values()) else 'FAIL_LOCAL_ATTENTION_ORIGINAL_TOLERANCE'
            self.steps.append(item)
            (self.output/(prefix+'-result.json')).write_text(json.dumps(item,indent=2)+'\n')
            assert item['status'].startswith('PASS'), item['status']
        finally:
            torch.ops._xpu_C.gdn_attention=original_gdn
            fa.flash_attn_varlen_func = guarded
            for key,value in saved_cells.items():
                cells[key].cell_contents = value
            state.restore()
            assert state.verify() == state.initial_digest
        return function(*args, **kwargs)


def install(runner, output):
    assert type(runner).__name__ == 'XPUModelRunnerV2'
    original = runner.execute_model
    assert hashlib.sha256(Path(inspect.getfile(inspect.unwrap(original.__func__))).read_bytes()).hexdigest() == RUNNER_SHA256
    assert runner.batch_sharder is None and runner.ubatch_runner is None and runner.pcp_manager is None and runner.dp_size == 1
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    sites = {'self.cudagraph_manager.run_fullgraph':0,'self.cudagraph_manager.run_pw_graph':0,'self.model':0}
    class Calls(ast.NodeTransformer):
        def visit_Call(self,node):
            node = self.generic_visit(node)
            name = ast.unparse(node.func)
            if name not in sites:
                return node
            sites[name] += 1
            return ast.copy_location(ast.Call(func=ast.Name(id='_matched_state_call',ctx=ast.Load()),
                args=[ast.Name(id=n,ctx=ast.Load()) for n in ('self','input_batch','batch_desc','model_inputs','attn_metadata','slot_mappings_by_layer')]+[node.func,*node.args],keywords=node.keywords),node)
    tree = Calls().visit(tree)
    assert sites == dict.fromkeys(sites,1), sites
    namespace = dict(inspect.unwrap(original.__func__).__globals__)
    def call(r,b,d,i,m,s,f,*args,**kwargs):
        return r._matched_state_replay.call(b,d,i,m,s,f,args,kwargs)
    namespace['_matched_state_call'] = call
    exec(compile(ast.fix_missing_locations(tree),'<matched-state-V2>', 'exec'),namespace)
    runner.execute_model = types.MethodType(namespace[original.__name__],runner)
    runner._matched_state_replay = Replay(runner,output)
    return dict(source_sha256=RUNNER_SHA256,call_sites=sites,scope='Diagnostic only, target body replay from saved active KV/GDN blocks; full-vocabulary head on actual batch rows.')


class ReplayWorkerExtension:
    def install_matched_state(self,output):
        return install(self.model_runner,output)
    def arm_matched_state(self,case):
        self.model_runner._matched_state_replay.arm(case)
    def finish_matched_state(self):
        replay = self.model_runner._matched_state_replay
        replay.enabled = False
        assert not replay.wanted, f'Missing actual batch shapes: {replay.wanted}'
        return [dict(case=s['case'],label=s['label'],status=s['status'],graph_mode=s['graph_mode'],
                     actual_rows=s['actual_rows'],rows_by_request=s['rows_by_request']) for s in replay.steps]

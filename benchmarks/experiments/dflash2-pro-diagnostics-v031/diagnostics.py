"""Read-only worker diagnostics. Tensor capture is a separate, perturbed wave."""
import copy
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
import sys
import types

import torch
from wide_inspect import WorkerExtension as Attestation
from profile_capture import ProfileWorkerExtension


def plain(value):
    if dataclasses.is_dataclass(value):
        return {"class": type(value).__name__, **{
            f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def cpu(value):
    return value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value


class WorkerExtension(Attestation, ProfileWorkerExtension):
    def rejection_statistical_test(self):
        from rejection_reference import gpu_test
        result = gpu_test()
        Path('/results/rejection-statistics.json').write_text(json.dumps(result, indent=2)+'\n')
        return result

    def cache_ledger(self):
        runner = self.model_runner
        cfg = runner.vllm_config
        cache = runner.kv_cache_config
        from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
        allocations = [plain(t) for t in cache.kv_cache_tensors]
        # KVCacheTensor descriptors deliberately alias one device pool. Dedup
        # actual backing storages; summing descriptor.size overcounts groups.
        storages = {}
        for tensor in runner.kv_caches:
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        total = sum(storages.values())
        assert cache.num_blocks > 0 and total % cache.num_blocks == 0
        pool_stride = total // cache.num_blocks
        contexts = {}
        for length in (96000, 192000):
            hypothetical = copy.copy(cfg)
            hypothetical.model_config = copy.copy(cfg.model_config)
            hypothetical.model_config.max_model_len = length
            groups = []
            for index, group in enumerate(cache.kv_cache_groups):
                spec = group.kv_cache_spec
                memory = spec.max_memory_usage_bytes(hypothetical)
                pages = (memory + spec.page_size_bytes - 1) // spec.page_size_bytes
                inner = getattr(spec, "kv_cache_specs", None)
                layers = inner or {name: spec for name in group.layer_names}
                groups.append(dict(index=index, layer_names=group.layer_names,
                    is_eagle_group=group.is_eagle_group, host_resident=group.host_resident,
                    spec=plain(spec), page_size_bytes=spec.page_size_bytes,
                    max_request_bytes=memory, resident_pool_blocks_per_request=pages,
                    padding_bytes_per_resident_pool_block=pool_stride-sum(s.page_size_bytes for s in layers.values()),
                    block_table_entries=spec.max_num_blocks_per_req(hypothetical, length),
                    layers={name: dict(spec=plain(s), page_size_bytes=s.page_size_bytes,
                        max_request_bytes=s.max_memory_usage_bytes(hypothetical))
                        for name, s in layers.items()}))
            blocks = sum(g['resident_pool_blocks_per_request'] for g in groups if not g['host_resident'])
            reported, concurrency = get_kv_cache_capacity(hypothetical, cache)
            contexts[str(length)] = dict(groups=groups, pool_blocks_per_request=blocks,
                required_shared_pool_bytes=blocks * pool_stride,
                worst_case_layer_padding_bytes=sum(g['resident_pool_blocks_per_request']*g['padding_bytes_per_resident_pool_block'] for g in groups),
                derived_cache_tokens=reported, derived_max_concurrency=concurrency,
                scope="Current group specs and pool stride, changed max_model_len only; CPU model, not actual capacity qualification")
        sources = {}
        destination = Path('/results/runtime-sources')
        for name, module in list(sys.modules.items()):
            if not name.startswith(('vllm.', 'exl3xpu.')):
                continue
            source = getattr(module, '__file__', None)
            if source is None or not source.endswith('.py'):
                continue
            p = Path(source)
            if not p.is_file():
                continue
            data = p.read_bytes()
            relative = name.replace('.', '/') + '.py'
            out = destination / relative
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            sources[name] = dict(path=source, sha256=hashlib.sha256(data).hexdigest(), snapshot=relative)
        required = ['vllm.model_executor.models.qwen3_next',
                    'vllm.model_executor.models.interfaces',
                    'vllm.model_executor.layers.layernorm']
        assert all(name in sources for name in required), required
        free, device_total = torch.xpu.mem_get_info()
        draft = runner.speculator
        buffers = {name: dict(shape=list(value.shape), dtype=str(value.dtype),
                   logical_bytes=value.numel()*value.element_size())
                   for name, value in vars(draft).items() if isinstance(value, torch.Tensor)}
        result = dict(schema=1, status='ACTUAL_WORKER_LEDGER_CPU_CONTEXT_MODEL',
            configured_max_model_len=cfg.model_config.max_model_len,
            num_pool_blocks=cache.num_blocks, pool_stride_bytes=pool_stride,
            total_device_kv_allocation_bytes=total, allocation_descriptors=allocations,
            actual_unique_kv_storages=[dict(device=k[0], bytes=v) for k, v in storages.items()],
            contexts=contexts, speculator_buffers=buffers,
            torch_allocated_bytes=torch.xpu.memory_allocated(),
            torch_reserved_bytes=torch.xpu.memory_reserved(),
            device_free_bytes=free, device_total_bytes=device_total,
            graph_scratch_scope='Included in device/allocator residency; not independently attributed by this ledger',
            sources=sources,
            prefix_marker_consumer='KVCacheCoordinator.eagle_group_ids; empty DFlash markers conservatively flag all groups for last-block drop')
        Path('/results/cache-ledger.json').write_text(json.dumps(result, indent=2)+'\n')
        return {key: value for key, value in result.items() if key not in ('sources', 'contexts', 'allocation_descriptors')}

    def install_round_capture(self, selected=(2, 8, 16, 32, 48, 64, 80, 96)):
        runner = self.model_runner
        assert hasattr(runner, '_exl3_event_capture')
        assert not runner._exl3_event_capture.enabled
        rejection = runner.rejection_sampler
        assert rejection.synthetic_conditional_rates is None
        assert not rejection.use_block_verification
        assert not runner.speculator.enable_adaptive_verification
        original = rejection._verify
        signature = inspect.signature(original)
        assert list(signature.parameters) == ['logits', 'draft_logits', 'draft_sampled', 'pos',
            'cu_num_logits', 'idx_mapping', 'idx_mapping_np', 'expanded_idx_mapping',
            'expanded_local_pos', 'seq_lens_upper_bound_np']
        state = dict(enabled=False, rounds=0, selected=set(selected), files=[])
        folder = Path('/results/round-capture'); folder.mkdir(exist_ok=True)

        def verify(*args, **kwargs):
            bound = signature.bind(*args, **kwargs).arguments
            record = None
            if state['enabled']:
                state['rounds'] += 1
                if state['rounds'] in state['selected']:
                    assert len(bound['idx_mapping_np']) == 1, 'Capture fixture requires C1'
                    idx = bound['idx_mapping_np'].tolist()
                    record = {name: cpu(value) for name, value in bound.items() if name not in ('logits', 'draft_logits')}
                    record['draft_logits'] = cpu(bound['draft_logits'][idx]) if bound['draft_logits'] is not None else None
                    sampling = runner.sampler.sampling_states
                    record.update(temperature=cpu(sampling.temperature.gpu[idx]),
                        seeds=cpu(sampling.seeds.gpu[idx]),
                        candidate_ids=cpu(runner.speculator.candidate_sampler.cached_candidate_ids[idx]) if bound['draft_logits'] is not None else None,
                        realized_scores=cpu(runner.speculator.candidate_sampler.scores[:1]),
                        proposed_tokens=cpu(runner.speculator.draft_tokens[:1]),
                        round=state['rounds'], mask_token_id=248070,
                        draft_sample_method=runner.vllm_config.speculative_config.draft_sample_method,
                        scope='Actual q before rejection and transformed target logits after sampling parameters; tensor copies perturb this wave')
            value = original(*args, **kwargs)
            if record is not None:
                record.update(processed_target_logits=cpu(value[0]),
                    sampled_token_ids=cpu(value[1]), num_sampled=cpu(value[2]))
                path = folder/f"round-{state['rounds']:04}.pt"
                torch.save(record, path); state['files'].append(str(path))
            return value

        rejection._verify = verify
        runner._dflash_round_capture = state
        return dict(installed=True, selected=list(selected),
                    limitations=['Graph-internal unary/edges/branches and paged K/V not yet captured; this is not the complete A1 gate'])

    def begin_round_capture(self):
        state = self.model_runner._dflash_round_capture
        assert not state['enabled'] and state['rounds'] == 0
        state['enabled'] = True
        return dict(enabled=True)

    def finish_round_capture(self):
        state = self.model_runner._dflash_round_capture
        state['enabled'] = False
        result = dict(rounds=state['rounds'], files=state['files'],
                      selected=sorted(state['selected']), status='PARTIAL_A1_CAPTURE')
        Path('/results/round-capture/index.json').write_text(json.dumps(result, indent=2)+'\n')
        return result

"""vLLM 0.31 admission policy: reserve full input AND bounded output before admission.

No live KV is swapped here. Native OffloadingConnector handles cache transfers.
The native hybrid coordinator remains the authority for block/page accounting.
"""
from vllm.v1.core.sched.async_scheduler import AsyncScheduler


def completion_target(request, model_limit, lookahead=0):
    return min(model_limit, request.num_prompt_tokens + request.max_tokens + lookahead)


class CompletionReservedScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        original = self.kv_cache_manager.allocate_slots

        def allocate(request, num_new_tokens, **options):
            admitting = not any(r.request_id == request.request_id for r in self.running)
            if admitting:
                reserve = self._completion_reservation(request, options)
                if reserve > self.kv_cache_manager.block_pool.get_num_free_blocks():
                    return None
            blocks = original(request, num_new_tokens, **options)
            if admitting and blocks is not None:
                self._record_gpu_prefix_reuse(request, options.get('num_new_computed_tokens', 0))
            return blocks

        self.kv_cache_manager.allocate_slots = allocate

    def _record_gpu_prefix_reuse(self, request, hit_tokens):
        """Keep CPU replacement policy informed of GPU hits in every hybrid group.

        Native store callbacks can skip the already cached recurrent tail. ARC
        then retains attention pages but evicts the recurrent checkpoint needed
        to resume them. This records reuse of *existing ready* CPU copies only;
        it neither allocates cache entries nor claims that a transfer occurred.
        """
        if not hit_tokens:
            return
        connector = getattr(self.connector, 'connector_scheduler', None)
        if connector is None:
            return
        status = connector._req_status.get(request.request_id)
        if status is None:
            return
        from vllm.v1.kv_offload.base import LookupResult
        keys = []
        for config, state in zip(connector.config.kv_group_configs, status.group_states):
            end = hit_tokens // config.tokens_per_chunk
            window = config.load_window_size_in_chunks(hit_tokens)
            start = 0 if window is None else max(0, end - window)
            keys.extend(key for key in state.offload_keys[start:end]
                        if connector.manager.lookup(key, status.req_context) == LookupResult.HIT)
        connector.manager._record_request_cache_access(keys, status.req_context, reused_keys=keys)

    def _blocks_remaining(self, request, options=None):
        options = options or {}
        manager = self.kv_cache_manager
        computed = options.get('new_computed_blocks')
        computed = computed.blocks if computed is not None else manager.empty_kv_cache_blocks.blocks
        local = request.num_computed_tokens + options.get('num_new_computed_tokens', 0)
        total = min(local + options.get('num_external_computed_tokens', 0), self.max_model_len)
        target = completion_target(request, self.max_model_len, self.num_lookahead_tokens)
        return manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=target,
            new_computed_blocks=computed,
            num_encoder_tokens=options.get('num_encoder_tokens', 0),
            total_computed_tokens=total,
            num_local_computed_tokens=local,
            num_tokens_main_model=target,
            apply_admission_cap=True,
            prefill_end=max(request.num_prompt_tokens, request.num_tokens - 1),
        )

    def _completion_reservation(self, request, options):
        # Include async loads as well as active decode/prefill; never double count.
        holders = {r.request_id: r for r in [*self.running, *self._inflight_prefills]
                   if r.request_id != request.request_id}
        remaining = sum(self._blocks_remaining(r) for r in holders.values())
        watermark = self.kv_cache_manager.watermark_blocks if holders else 0
        return remaining + self._blocks_remaining(request, options) + watermark

"""Bound XPU cache-restore batches on the pinned vLLM 0.33 runtime.

Shared mmap allocation and XPU host registration are now upstream features.
Registration can fall back to pageable memory, whose H2D copies may allocate
pinned staging. Keep both the normal and expedited load queues in at most
128 MiB batches, draining each batch before another staging allocation.
The native allocator, registration, transfer ordering and cache policy remain
upstream; no whole-history-sized staging allocation is needed on fallback.
"""
from functools import partial

from vllm.platforms import current_platform
from vllm.logger import init_logger
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


MAX_LOAD_BATCH_BYTES = 128 * 1024**2


def bounded_copy_ranges(sizes, limit=MAX_LOAD_BATCH_BYTES):
    """Partition CPU copy descriptors by payload, rejecting unbounded entries."""
    start = total = 0
    for index, size in enumerate(sizes):
        if size < 0 or size > limit:
            raise ValueError('XPU offload descriptor exceeds bounded staging budget')
        if total + size > limit:
            yield start, index
            start, total = index, 0
        total += size
    if total:
        yield start, len(sizes)


def bounded_load(copy, synchronize, src, dst, sizes, **kwargs):
    for start, end in bounded_copy_ranges(sizes.tolist()):
        copy(src[start:end], dst[start:end], sizes[start:end], **kwargs)
        # The native caching host allocator may recycle staging only after
        # its recorded DMA event completes. Merely slicing an async batch
        # would still accumulate all staging allocations at once.
        synchronize()


class BoundedXPUOffloadingSpec(CPUOffloadingSpec):
    def __init__(self, config):
        super().__init__(config)
        init_logger(__name__).info(
            'CPU_KV_CACHE_LAYOUT budget_bytes=%d allocated_bytes=%d chunk_bytes=%d chunks=%d groups=%s',
            self.extra_config['cpu_bytes_to_use'], self.num_chunks*self.kv_bytes_per_chunk,
            self.kv_bytes_per_chunk, self.num_chunks,
            [(g.group_id,len(g.layer_names),g.tokens_per_block) for g in config.groups])

    def create_worker(self, kv_caches):
        worker = super().create_worker(kv_caches)
        if current_platform.is_xpu():
            # vLLM 0.33 has independent normal and expedited H2D queues.
            # Both can restore a whole history and need bounded staging.
            for handler in (worker._load_handler, worker._expedited_load_handler):
                handler._swap_blocks_batch = partial(
                    bounded_load, handler._swap_blocks_batch,
                    lambda: current_platform.current_stream().synchronize())
        return worker

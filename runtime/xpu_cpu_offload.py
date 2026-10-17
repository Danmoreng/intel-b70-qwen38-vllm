"""Bound the XPU CPU cache with vLLM's mmap allocator instead of pinned pools.

PyTorch XPU's per-tensor pinned allocations round up and retain their backing
allocations. A logical 10 GiB cache can therefore consume substantially more
physical RAM. The upstream shared-region layout allocates exactly the budget
(rounded down to complete pages). XPU uses the upstream DMA transfer path;
CUDA host registration and the CUDA-only UVA kernels remain disabled.

Large XPU H2D batches additionally need a pinned staging allocation. Bound
those batches and drain each one before submitting the next, so restoring a
full history cannot allocate another full-history-sized host buffer.

This production adapter targets only the pinned vLLM 0.31.0 runtime.
"""
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

    def _uses_shared_region(self):
        return current_platform.is_xpu() or super()._uses_shared_region()

    def create_worker(self, kv_caches):
        worker = super().create_worker(kv_caches)
        if current_platform.is_xpu():
            handler = worker._load_handler
            original = handler._swap_blocks_batch

            def copy(src, dst, sizes, **kwargs):
                return bounded_load(original, current_platform.current_stream().synchronize,
                                    src, dst, sizes, **kwargs)

            handler._swap_blocks_batch = copy
        return worker

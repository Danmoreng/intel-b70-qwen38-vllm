"""Preserve complete dense-prefix alias views in native block COW copies.

F8 hybrid groups expose65 views on17 layer slots in one allocation. A Mamba
view has3,268,608 bytes/page; its aliased attention view has4,194,304. Native
pointer-only dedup kept the narrower first view, dropping valid attention KV.
Choose a covering dense prefix once per pointer, then use UNCHANGED native
copy code. No new stream, synchronization, memory allocation or inference op.
"""
import hashlib,inspect


def view_key(cache):
    return (cache.dtype,tuple(cache.shape),tuple(cache.stride()))


def dense_row_bytes(cache,num_blocks):
    ratio,remainder=divmod(cache.shape[0],num_blocks)
    if remainder:raise ValueError('Kernel block count not divisible by scheduler blocks')
    row=cache.unflatten(0,(num_blocks,ratio))
    size=cache.element_size()
    for stride,dim in sorted((s*cache.element_size(),d) for s,d in zip(row.stride()[1:],row.shape[1:]) if d>1):
        if stride!=size:raise ValueError('Different aliased KV views require dense per-block byte prefixes')
        size*=dim
    return size,row.stride(0)*row.element_size()


def complete_alias_views(caches,num_blocks):
    groups={}
    for cache in caches:groups.setdefault((cache.device,cache.data_ptr()),[]).append(cache)
    selected=[]
    for aliases in groups.values():
        unique={view_key(c):c for c in aliases}
        if len(unique)==1:selected.append(next(iter(unique.values())));continue
        candidates=[(dense_row_bytes(c,num_blocks),c) for c in unique.values()]
        if len({shape[1] for shape,c in candidates})!=1:
            raise ValueError('Aliased KV views use different scheduler-block byte strides')
        span,cover=max(candidates,key=lambda pair:pair[0][0])
        if span[0]>span[1]:raise ValueError('KV row extends past its scheduler-block stride')
        selected.append(cover)
    return selected


def install(r):
    import vllm.v1.worker.utils as utils
    import vllm.v1.worker.gpu.model_runner as runner
    native=utils.copy_kv_cache_blocks_inplace
    source=inspect.getsource(native)
    expected='key = (cache.device, cache.data_ptr())'
    assert expected in source and 'copied_storages' in source,'Unexpected native COW source; review before applying'
    assert hashlib.sha256(source.encode()).hexdigest()=='cf7a8c14bc451c966135277c443fac5df65c98c049e2dedcf207025f4b231bca','Native COW source hash differs from audited F8 runtime'
    assert runner.copy_kv_cache_blocks_inplace is native,'COW hook already installed or unexpected runner binding'
    def copy(caches,num_blocks,pairs):
        if not pairs:return
        return native(complete_alias_views(caches,num_blocks),num_blocks,pairs)
    # Keep the native utility available for audit/CPU regression proofs.
    runner.copy_kv_cache_blocks_inplace=copy
    raw=list(r.kv_caches);selected=complete_alias_views(raw,r.kv_cache_config.num_blocks)
    report=dict(status='COMPLETE_DENSE_ALIAS_COW_VIEW_SELECTION_INSTALLED',original_views=len(raw),selected_views=len(selected),
        native_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        patch_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),
        selected=[dict(shape=list(c.shape),stride=list(c.stride()),dtype=str(c.dtype),row_bytes=dense_row_bytes(c,r.kv_cache_config.num_blocks)[0]) for c in selected],
        scope='COW metadata/view selection only. Native copy, streams/events, attention, model kernels and sampling unchanged. No adoption without full numerical/cost gates.')
    r._complete_cow_copy_fix=report
    return report

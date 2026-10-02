"""Diagnostic-only cache observations from the actual prefill thread/queue.

Mounted only in a separate qualification worker. Never installed in the image
or used for the throughput matrix/matched performance comparison.
"""
import os

if os.environ.get('B70_REVIEW_CACHE_DIAGNOSTIC')=='1':
    import functools
    import json
    from pathlib import Path
    import threading
    import time
    import torch
    from exl3xpu import fp8kv_prefill

    original=fp8kv_prefill.prefill_attention_onednn
    previous={}

    @functools.wraps(original)
    def observed(E,q,key_cache,value_cache,pages,seq_len,k_scale,v_scale,scale,out):
        result=original(E,q,key_cache,value_cache,pages,seq_len,k_scale,v_scale,scale,out)
        # Called on the same inference thread and current queue, after the
        # actual native SDPA enqueue. The operation reads host cache metadata;
        # it does not wait for completion or copy device tensor values.
        stats=dict(torch.ops.exl3xpu_C.exl3_sdpa_cache_stats(q))
        stream=str(torch.xpu.current_stream(q.device))
        owner=(os.getpid(),threading.get_ident(),stream)
        signature=(stats['misses'],stats['evictions'],stats['hits']//128)
        if previous.get(owner)!=signature:
            previous[owner]=signature
            rss=int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))*1024
            record=dict(pid=owner[0],thread_id=owner[1],stream=stream,monotonic=time.monotonic(),
                exact_kv_tokens=seq_len,query_rows=q.shape[0],cache=stats,rss_bytes=rss,
                xpu_allocated_bytes=torch.xpu.memory_allocated(q.device),
                xpu_reserved_bytes=torch.xpu.memory_reserved(q.device),
                scope='Actual serving prefill thread/queue; instrumented diagnostic, not throughput')
            with Path('/results/cache-observations.jsonl').open('a') as log:
                log.write(json.dumps(record)+'\n')
        return result

    fp8kv_prefill.prefill_attention_onednn=observed

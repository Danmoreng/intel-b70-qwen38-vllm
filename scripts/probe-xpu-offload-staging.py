#!/usr/bin/env python3
"""Run inside the pinned XPU image, with the GPU otherwise idle. No model loaded."""
import json
import mmap
from pathlib import Path
import threading
import time

import torch
from vllm import _custom_ops as ops
from xpu_cpu_offload import bounded_load, MAX_LOAD_BATCH_BYTES


def available():
    return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def main():
    page=3276800
    count=2800
    payload=page*count
    cache_bytes=12*1024**3
    if available()<cache_bytes+6*1024**3:
        raise RuntimeError('Insufficient headroom for isolated staging probe')
    samples=[];done=threading.Event()
    def observe():
        while not done.wait(.05):samples.append(available())
    thread=threading.Thread(target=observe,daemon=True);thread.start()
    try:
        region=mmap.mmap(-1,cache_bytes)
        source=torch.frombuffer(region,dtype=torch.int8)
        source.fill_(37)
        target=torch.zeros(payload,dtype=torch.int8,device='xpu')
        torch.xpu.synchronize()
        src=torch.tensor([source.data_ptr()+i*page for i in range(count)],dtype=torch.uint64)
        dst=torch.tensor([target.data_ptr()+i*page for i in range(count)],dtype=torch.uint64)
        sizes=torch.full((count,),page,dtype=torch.uint64)
        start=time.monotonic()
        bounded_load(ops.swap_blocks_batch,torch.xpu.current_stream().synchronize,
                     src,dst,sizes,is_src_access_order_any=True)
        # Inspect only a few bytes per GPU page; copying the whole tensor back
        # to CPU would defeat the memory bound this probe is checking.
        for offset in [0,page//2,page-1]:
            if not bool(torch.all(target[offset::page]==37)):
                raise RuntimeError('Large bounded transfer corrupted data')
        print(json.dumps({'validation':'passed','cpuCacheBytes':cache_bytes,
            'payloadBytes':payload,'maxStagingBatchBytes':MAX_LOAD_BATCH_BYTES,
            'seconds':time.monotonic()-start,
            'minHostMemAvailableBytes':min(samples) if samples else available(),
            'cgroupMemoryPeakBytes':int(Path('/sys/fs/cgroup/memory.peak').read_text())},indent=2),flush=True)
    finally:
        done.set();thread.join(timeout=2)


if __name__=='__main__':main()

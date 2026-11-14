"""Retire processed GDN states across null holes; preserve in-flight boundaries.

The pinned old allocator stopped at the first null entry. GDN tables are sparse
and can contain older live states to the left of that hole, accumulating one
orphan per prefill chunk. Only the Mamba manager receives this override.
"""
import hashlib
import inspect
import os
from pathlib import Path
from vllm.logger import init_logger
logger=init_logger('vllm.exl3xpu.cache')
OLD_SOURCE_SHA256='bcb27e38895332bf6a4c55608f2917eb9fd941ec5a629c14b88358f00aadeba8'

def retire_sparse_states(self,request_id,first_block,last_block):
    if request_id not in self.req_to_blocks or first_block>=last_block: return
    blocks=self.req_to_blocks[request_id]
    last_block=min(last_block,len(blocks))
    freed=[]
    for i in range(last_block-1,first_block-1,-1):
        if blocks[i]==self._null_block: continue
        freed.append(blocks[i]);blocks[i]=self._null_block
    if freed: self.block_pool.free_blocks(freed)

def install():
    if os.environ.get('EXL3_FIX_SPARSE_GDN_RETIREMENT','0')!='1': return False
    from vllm.v1.core import single_type_kv_cache_manager as module
    cls=module.MambaManager
    if getattr(cls,'_exl3_sparse_retirement_fixed',False): return True
    sha=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if sha!=OLD_SOURCE_SHA256: raise RuntimeError(f'Unknown GDN allocator for retirement fix: {module.__file__}/{sha}')
    cls._remove_blocks_in_range=retire_sparse_states
    cls._exl3_sparse_retirement_fixed=True
    logger.info('EXL3 GDN retirement fix: scan past null holes within the existing processed-token boundary; source=%s',sha)
    return True

"""Sparse GDN retirement: no orphan states, no premature reuse, no double free."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from exl3xpu.cache_retirement import retire_sparse_states,install
from vllm.v1.core.single_type_kv_cache_manager import MambaManager,FullAttentionManager
ORIGINAL_RETIRE=MambaManager._remove_blocks_in_range

class Block:
    def __init__(self,n,refs=1): self.block_id=n;self.ref_cnt=refs

class Pool:
    def __init__(self): self.freed=[]
    def free_blocks(self,blocks):
        for b in blocks:
            if b.ref_cnt<=0: raise AssertionError('double free')
            b.ref_cnt-=1;self.freed.append(b.block_id)

class RetirementTests(unittest.TestCase):
    def manager(self,blocks):
        null=Block(0,0);values=[null if b is None else Block(b) for b in blocks]
        return SimpleNamespace(req_to_blocks={'request':values},_null_block=null,block_pool=Pool())
    def test_old_sparse_hole_regression(self):
        m=self.manager([1,None,2,None,3])
        ORIGINAL_RETIRE(m,'request',0,4)
        self.assertEqual(m.block_pool.freed,[]) # demonstrate the pinned original bug
        retire_sparse_states(m,'request',0,4)
        self.assertEqual(m.block_pool.freed,[2,1]);self.assertEqual(m.req_to_blocks['request'][4].block_id,3)
    def test_processed_boundary_preserves_current_and_speculative_states(self):
        m=self.manager([1,None,None,2,None,None,None,3,4,5,6])
        retire_sparse_states(m,'request',0,7)
        self.assertEqual(m.block_pool.freed,[2,1])
        self.assertEqual([b.block_id for b in m.req_to_blocks['request'][7:]],[3,4,5,6])
    def test_shared_state_keeps_other_reference(self):
        m=self.manager([1,None,2]);m.req_to_blocks['request'][0].ref_cnt=2
        state=m.req_to_blocks['request'][0]
        retire_sparse_states(m,'request',0,2);self.assertEqual(state.ref_cnt,1)
    def test_repeated_retirement_and_bounds(self):
        m=self.manager([1,None,2]);retire_sparse_states(m,'request',0,99);retire_sparse_states(m,'request',0,99)
        self.assertEqual(m.block_pool.freed,[2,1])
        retire_sparse_states(m,'unknown',0,99);retire_sparse_states(m,'request',3,2)
    def test_install_only_changes_mamba(self):
        original=FullAttentionManager._remove_blocks_in_range
        with patch.dict(os.environ,{'EXL3_FIX_SPARSE_GDN_RETIREMENT':'1'}):
            self.assertTrue(install());self.assertTrue(install())
        self.assertIs(FullAttentionManager._remove_blocks_in_range,original)
        self.assertIs(MambaManager._remove_blocks_in_range,retire_sparse_states)

if __name__=='__main__': unittest.main()

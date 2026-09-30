"""A replay must preserve KV selection, tensor aliases and local pointer tables."""
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
    import vllm
except ImportError as error:
    raise unittest.SkipTest('Requires the pinned B70 image') from error

spec = importlib.util.spec_from_file_location('cycle_replay', Path(__file__).parents[2] / 'scripts/b70_cycle_replay.py')
cycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cycle)


@dataclass(frozen=True)
class CacheSpec:
    block_size: int = 1664


class State:
    __module__ = 'vllm.snapshot_fixture'

    def __init__(self, **fields):
        self.__dict__.update(fields)


def runner(capacity=7, pointer=100):
    pool = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    empty = State()
    cache = torch.arange(capacity * 8, dtype=torch.float32).reshape(capacity, 2, 4)
    return State(req_states=State(pool=pool, gpu=pool[0], _curr=0, mapping={'a': 0}, req_id_to_index={'a': 0, 'b': 1}, optional=None),
        input_buffers=State(input_ids=torch.tensor([7, 8])),
        block_tables=State(block_tables=[State(gpu=torch.tensor([[1, 3], [2, 4]]))],
            num_blocks=State(np=__import__('numpy').array([[2, 2]])),
            block_table_ptrs=torch.tensor([pointer], dtype=torch.int64)),
        model_state=State(_mamba_spec=CacheSpec()), sampler=empty, speculator=empty, cudagraph_manager=empty,
        draft_tokens_handler=empty, attn_groups=[], max_num_reqs=2,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=['layer'])]),
        vllm_config=SimpleNamespace(compilation_config=SimpleNamespace(
            static_forward_context={'layer': SimpleNamespace(kv_cache=cache)})))


class Snapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cycle.install()

    def test_restore_alias_cpu_bookkeeping_and_active_kv(self):
        a = runner()
        before, refs = cycle.snapshot_capture(a)
        old_alias = a.req_states.gpu
        a.req_states.gpu = a.req_states.pool[1]
        a.req_states._curr = 1
        a.req_states.mapping['new'] = 1
        a.req_states.optional = State(new_buffer=torch.ones(3))
        a.req_states.pool[0].zero_()
        cache = a.vllm_config.compilation_config.static_forward_context['layer'].kv_cache
        cache.zero_()
        cycle.snapshot_restore(a, before, refs)
        self.assertIs(a.req_states.gpu, old_alias)
        self.assertEqual(a.req_states._curr, 0)
        self.assertEqual(a.req_states.mapping, {'a': 0})
        self.assertIsNone(a.req_states.optional)
        self.assertTrue(torch.equal(old_alias, torch.tensor([1, 2])))
        self.assertTrue(torch.equal(cache[:5], torch.arange(56.).reshape(7, 2, 4)[:5]))
        self.assertEqual(float(cache[6].sum()), 0)  # unreferenced block excluded

    def test_canonical_state_cross_capacity_keeps_local_pointers(self):
        a, b = runner(7, 100), runner(6, 999)
        before, _ = cycle.snapshot_capture(a)
        b.input_buffers.input_ids.zero_()
        b.vllm_config.compilation_config.static_forward_context['layer'].kv_cache.fill_(-1)
        cycle.snapshot_restore(b, before)
        self.assertEqual(b.block_tables.block_table_ptrs.item(), 999)
        self.assertTrue(torch.equal(b.input_buffers.input_ids, torch.tensor([7, 8])))
        cache = b.vllm_config.compilation_config.static_forward_context['layer'].kv_cache
        self.assertTrue(torch.equal(cache[:5], torch.arange(56.).reshape(7, 2, 4)[:5]))
        self.assertEqual(float(cache[5].sum()), -8)


if __name__ == '__main__':
    with patch.object(torch.xpu, 'synchronize'), patch.object(torch.xpu, 'get_rng_state', return_value=torch.get_rng_state()), patch.object(torch.xpu, 'set_rng_state'):
        unittest.main()

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch


class NativeScheduler:
    pass


spec = importlib.util.spec_from_file_location('kv_admission', Path(__file__).resolve().parents[2]/'runtime/kv_admission.py')
module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'vllm.v1.core.sched.async_scheduler': SimpleNamespace(AsyncScheduler=NativeScheduler)}):
    spec.loader.exec_module(module)


def request(name, prompt, output, computed=0):
    return SimpleNamespace(request_id=name, num_prompt_tokens=prompt, max_tokens=output,
                           num_tokens=prompt, num_computed_tokens=computed)


class AdmissionTests(unittest.TestCase):
    def policy(self, running=(), loading=(), free=293, allocated=None):
        allocated = allocated or {}
        policy = object.__new__(module.CompletionReservedScheduler)
        policy.max_model_len = 262144
        policy.num_lookahead_tokens = 4
        policy.running = list(running)
        policy._inflight_prefills = list(loading)
        def blocks(**args):
            # Fake coordinator reports group-specific missing blocks; the policy
            # must consume that result rather than assume one token == one page.
            return max(0, (args['num_tokens'] + 999)//1000 - allocated.get(args['request_id'], 0))
        policy.kv_cache_manager = SimpleNamespace(
            empty_kv_cache_blocks=SimpleNamespace(blocks=([], [])),
            coordinator=SimpleNamespace(get_num_blocks_to_allocate=blocks),
            block_pool=SimpleNamespace(get_num_free_blocks=lambda: free), watermark_blocks=15)
        return policy

    def test_two_200k_inputs_wait_instead_of_overadmitting_during_first_chunk(self):
        first = request('first', 200000, 16384, computed=3200)
        candidate = request('second', 200000, 16384)
        p = self.policy(running=[first], allocated={'first': 4}, free=289)
        self.assertGreater(p._completion_reservation(candidate, {}), 289)

    def test_short_parallel_requests_fit(self):
        p = self.policy(running=[request('one', 16000, 1024)], allocated={'one': 17}, free=276)
        self.assertLess(p._completion_reservation(request('two', 16000, 1024), {}), 276)

    def test_output_growth_is_reserved_and_capped_at_model_limit(self):
        self.assertEqual(module.completion_target(request('x', 250000, 16000), 262144, 4), 262144)
        self.assertEqual(module.completion_target(request('x', 200000, 16000), 262144, 4), 216004)

    def test_async_loads_are_reserved_once_and_current_request_is_excluded(self):
        first = request('one', 80000, 1000)
        candidate = request('two', 80000, 1000)
        p = self.policy(running=[first], loading=[first, candidate])
        self.assertEqual(p._completion_reservation(candidate, {}), 82 + 82 + 15)

    def test_single_maximum_request_does_not_deadlock_on_watermark(self):
        p = self.policy()
        self.assertEqual(p._completion_reservation(request('x', 246000, 16144), {}), 263)

    def test_gpu_hit_reuses_attention_prefix_and_only_needed_recurrent_tail(self):
        from unittest.mock import Mock
        p = self.policy()
        context = object()
        manager = SimpleNamespace(lookup=Mock(side_effect=lambda key, ctx: 'ready' if key != 'a1' else 'pending'),
                                  _record_request_cache_access=Mock())
        p.connector = SimpleNamespace(connector_scheduler=SimpleNamespace(
            manager=manager, config=SimpleNamespace(kv_group_configs=[
                SimpleNamespace(tokens_per_chunk=1600, load_window_size_in_chunks=lambda tokens: None),
                SimpleNamespace(tokens_per_chunk=1600, load_window_size_in_chunks=lambda tokens: 1)]),
            _req_status={'x': SimpleNamespace(req_context=context, group_states=[
                SimpleNamespace(offload_keys=['a0', 'a1', 'a2', 'a3']),
                SimpleNamespace(offload_keys=['m0', 'm1', 'm2', 'm3'])])}))
        with patch.dict(sys.modules, {'vllm.v1.kv_offload.base': SimpleNamespace(LookupResult=SimpleNamespace(HIT='ready'))}):
            p._record_gpu_prefix_reuse(request('x', 6400, 64), 4800)
        manager._record_request_cache_access.assert_called_once_with(
            ['a0', 'a2', 'm2'], context, reused_keys=['a0', 'a2', 'm2'])

    def test_no_offloader_or_no_gpu_hit_needs_no_cpu_policy_update(self):
        p = self.policy();p.connector = None
        p._record_gpu_prefix_reuse(request('x', 6400, 64), 4800)
        p._record_gpu_prefix_reuse(request('x', 6400, 64), 0)


if __name__ == '__main__': unittest.main()

"""CPU proof of mixed metadata and fail-closed semantic guards."""
import unittest
from types import SimpleNamespace

import torch

from exl3xpu.attention_metadata import host_segments, prepare
from exl3xpu.attention_dispatch import supported_features


class Contract(unittest.TestCase):
    def test_interleaved_and_true_per_sequence_maximum(self):
        p = host_segments([0, 4, 8, 264, 266, 269, 333],
                          [10000, 20000, 103001, 30000, 40000, 4097],
                          [False, False, True, False, False, True], 333,
                          fixed_partition=True)
        self.assertEqual([(s.first, s.last, s.maximum_query, s.maximum_kv, s.prefill)
                          for s in p.segments],
                         [(0, 2, 4, 20000, False), (2, 3, 256, 103001, True),
                          (3, 5, 3, 40000, False), (5, 6, 64, 4097, True)])
        # Legacy total-query maximum would incorrectly be 8 and 5.
        self.assertEqual(sum(s.stop-s.start for s in p.segments), 333)

    def test_thresholds_and_prefill_only_exact_lengths(self):
        for q in [1, 2, 3, 4, 5, 63, 64, 255, 256]:
            for length in [4095, 4096, 4097]:
                for flag in [False, True]:
                    plan = host_segments([0, q], [length], [flag], q, fixed_partition=True)
                    self.assertEqual(plan is not None, flag and q >= 64 and length >= 4096)

    def test_inexact_and_malformed_fallback(self):
        for bounds, lengths, flags, tokens, fixed in [
            ([0, 256], [4096], [True], 256, False),
            ([1, 257], [4096], [True], 256, True),
            ([0, 256, 256], [4096, 100], [True, False], 256, True),
            ([0, 256], [100], [True], 256, True),
            ([0, 256], [4096], [True], 260, True),
            ([0, 256], [4096], [1], 256, True),
        ]:
            self.assertIsNone(host_segments(bounds, lengths, flags, tokens, fixed_partition=fixed))

    def test_cpu_prepare_native_offsets(self):
        c = SimpleNamespace(max_query_len=256, num_reqs=4, num_actual_tokens=266,
            query_start_loc_cpu=torch.tensor([0, 4, 260, 263, 266], dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.tensor([10000, 4097, 10000, 15000], dtype=torch.int32),
            is_prefilling=torch.tensor([False, True, False, False]),
            query_start_loc=torch.tensor([0, 4, 260, 263, 266], dtype=torch.int32))
        p = prepare(c, fixed_partition=True)
        self.assertEqual(p.segments[-1].cu_query.tolist(), [0, 3, 6])
        self.assertEqual(p.segments[-1].maximum_query, 3)
        self.assertIsNone(prepare(c, fixed_partition=False))
        c.is_prefilling = None
        self.assertIsNone(prepare(c, fixed_partition=True))

    def test_semantic_native_fallbacks(self):
        d = dict(causal=True, softmax_scale=.0625, fa_version=2, window_size=None)
        self.assertTrue(supported_features(d))
        for k, value in [('causal', False), ('window_size', [128, -1]),
                         ('s_aux', object()), ('alibi_slopes', object()),
                         ('softcap', 1.), ('q_descale', object()),
                         ('return_softmax_lse', True), ('dynamic_causal', object()),
                         ('scheduler_metadata', object()), ('mask_mod', object()),
                         ('unknown_flag', False), ('fa_version', 4)]:
            self.assertFalse(supported_features(dict(d, **{k:value})), k)


if __name__ == '__main__':
    unittest.main()

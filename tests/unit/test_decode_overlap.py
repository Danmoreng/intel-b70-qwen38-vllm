import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parents[2] / 'scripts'))
from decode_overlap import pool_overlap, summarize_overlap


def sample(t, d, a, y, **changes):
    return dict(elapsed_s=t, draft_tokens=d, accepted_tokens=a,
                generation_tokens=y, running=4, waiting=0,
                prefill_computed=100, **changes)


class DecodeOverlap(unittest.TestCase):
    def test_uses_identical_endpoints_and_pools_rounds(self):
        samples = [sample(0, 0, 0, 0), sample(1, 32, 16, 24),
                   sample(2, 192, 96, 144), sample(4, 320, 160, 240)]
        row = summarize_overlap(samples, 1, 3, 4)
        self.assertTrue(row['fully_overlapped_counter_window_valid'])
        self.assertEqual(row['fully_overlapped_draft_tokens'], 160)
        self.assertEqual(row['fully_overlapped_speculative_acceptance'], .5)
        self.assertEqual(row['fully_overlapped_round_equivalent_ms'], 100)
        row['fully_overlapped_decode_sampled_s'] = 1
        second = {**row, 'fully_overlapped_decode_sampled_s': 2}
        self.assertEqual(pool_overlap([row, second])['fully_overlapped_round_equivalent_ms'], 150)

    def test_excludes_mixed_work_and_bad_accounting_from_round_time(self):
        for changes, issue in [({'running': 3}, 'nonconstant_occupancy'),
                               ({'prefill_computed': 101}, 'prefill_inside_window'),
                               ({'generation_tokens': 13}, 'token_accounting_mismatch')]:
            last = {**sample(2, 16, 8, 12), **changes}
            row = summarize_overlap([sample(1, 0, 0, 0), last], 1, 2, 4)
            self.assertIn(issue, row['fully_overlapped_counter_window_issues'])
            self.assertIsNone(row['fully_overlapped_round_equivalent_ms'])

    def test_insufficient_samples_and_scrape_error(self):
        self.assertFalse(summarize_overlap([], 1, 2, 4)['fully_overlapped_counter_window_valid'])
        row = summarize_overlap([sample(1, 0, 0, 0), {'elapsed_s': 1.5, 'error': 'timeout'},
                                 sample(2, 16, 8, 12)], 1, 2, 4)
        self.assertIn('metric_sample_error', row['fully_overlapped_counter_window_issues'])


if __name__ == '__main__':
    unittest.main()

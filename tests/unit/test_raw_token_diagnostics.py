"""Regression for the real tool-parser-hidden token1023 failure, offline only."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / 'benchmarks/experiments/vllm-033-upgrade/run-frozen-request-probe.py'
EVIDENCE = ROOT / 'benchmarks/results/direct-xpu-20261114/qualification/generation-probes/r11-raw'
spec = importlib.util.spec_from_file_location('frozen_request_probe', PROBE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class RawTokenDiagnostics(unittest.TestCase):
    def test_real_hidden_tool_loops(self):
        for case, start, length in [('selected-native', 885, 1163),
                                    ('selected-eager', 885, 1163),
                                    ('torch2141-new-intel', 694, 1354),
                                    ('isolated-request-cache', 692, 1356)]:
            with self.subTest(case=case):
                response = json.loads((EVIDENCE / case / 'response-002.json').read_text())
                ids = response['choices'][0]['token_ids']
                run = probe.token_run_summary(ids)['longest']
                self.assertEqual((run['start_index'], run['token_id'], run['length']),
                                 (start, 1023, length))
                self.assertEqual(len(ids), response['usage']['completion_tokens'])

    def test_real_successful_controls(self):
        for case in ('reference-native', 'no-mtp'):
            response = json.loads((EVIDENCE / case / 'response-002.json').read_text())
            self.assertEqual(probe.token_run_summary(response['choices'][0]['token_ids'])
                             ['longest']['length'], 1)

    def test_empty_is_distinct_from_missing_ids(self):
        self.assertEqual(probe.token_run_summary([]),
                         {'count': 0, 'longest': None, 'suffix': None})


if __name__ == '__main__':
    unittest.main()

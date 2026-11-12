import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('watchdog', Path(__file__).with_name('inference-watchdog.py'))
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def metrics(tokens=100, running=2):
    return dict(generation_tokens_total=tokens, prompt_tokens_total=1000,
                request_success_total=3, num_requests_running=running, num_requests_waiting=0)


class WatchdogTests(unittest.TestCase):
    def observe(self, state=None, now=1000, sample=None, healthy=True, fault=False, **kw):
        return w.observe(state or {}, 'invocation', 2000, now,
                         metrics() if sample is None else sample, healthy, fault, **kw)

    def test_operator_stopped_service_stays_stopped(self):
        with patch.object(w.subprocess, 'check_output', return_value='ActiveState=inactive\nInvocationID=\nExecMainStartTimestampMonotonic=0\n'), patch.object(w, 'fetch') as fetch:
            w.main()
            fetch.assert_not_called()

    def test_real_metrics_and_invalid_samples(self):
        sample = 'vllm:num_requests_running{engine="0"} 2\nvllm:generation_tokens_total{engine="0"} 100\nvllm:prompt_tokens_total{engine="0"} 1000\n'
        self.assertEqual(w.parse_metrics(sample), dict(num_requests_running=2, generation_tokens_total=100, prompt_tokens_total=1000))
        for invalid in ['', sample.replace('1000', 'NaN'), sample.replace(' 100\n', ' -1\n')]:
            with self.assertRaises(ValueError):
                w.parse_metrics(invalid)

    def test_confirmed_gpu_fault_and_hung_health_recovers(self):
        state, reason = self.observe(fault=True)
        self.assertIsNone(reason)
        state, reason = self.observe(state, now=1059)
        self.assertIsNone(reason)
        state, reason = self.observe(state, now=1060)
        self.assertIn('GPU', reason)
        self.assertEqual(state['restarts'], [1060])

    def test_long_prefill_is_allowed_and_progress_clears_fault(self):
        state, _ = self.observe()
        _, reason = self.observe(state, now=1599)
        self.assertIsNone(reason)
        _, reason = self.observe(state, now=1600)
        self.assertIn('10 minutes', reason)
        state, _ = self.observe(fault=True)
        state, reason = self.observe(state, now=1060, sample=metrics(101))
        self.assertIsNone(reason)
        self.assertFalse(state['gpu_fault'])

    def test_idle_and_starting_service_are_not_restarted(self):
        state, _ = self.observe(sample=metrics(running=0), fault=True)
        _, reason = self.observe(state, now=3000, sample=metrics(running=0))
        self.assertIsNone(reason)
        state, reason = w.observe({}, 'new', 200, 1000, None, False, True)
        self.assertIsNone(reason)
        _, reason = w.observe(state, 'new', 899, 1699, None, False, False)
        self.assertIsNone(reason)

    def test_new_invocation_resets_stall_but_keeps_restart_history(self):
        state, _ = self.observe(fault=True)
        state['restarts'] = [900]
        state, reason = w.observe(state, 'new', 2000, 2000, metrics(), True, False)
        self.assertIsNone(reason)
        self.assertEqual(state['progress_at'], 2000)
        self.assertEqual(state['restarts'], [900])

    def test_restart_cooldown_hourly_limit_and_daily_cutoff(self):
        state, _ = self.observe(fault=True)
        state['restarts'] = [950]
        _, reason = self.observe(state, now=1100)
        self.assertIsNone(reason)
        state['restarts'] = [100, 200, 300]
        _, reason = self.observe(state, now=1600)
        self.assertIsNone(reason)
        state['restarts'] = []
        _, reason = self.observe(state, now=1600, allowed=False)
        self.assertIsNone(reason)

    def test_unavailable_api_after_startup_grace(self):
        state, _ = w.observe({}, 'new', 900, 1000, None, False, False)
        _, reason = w.observe(state, 'new', 1500, 1600, None, False, False)
        self.assertIn('unavailable', reason)


if __name__ == '__main__':
    unittest.main()

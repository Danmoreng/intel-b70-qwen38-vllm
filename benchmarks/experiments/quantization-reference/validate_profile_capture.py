"""Check real V2 call-site guards and event plumbing without using a GPU."""
import json
from pathlib import Path
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import profile_capture as P
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class FakeEvent:
    clock = 0
    synchronizations = 0

    def __init__(self, enable_timing):
        assert enable_timing

    def record(self):
        self.timestamp = self.clock

    def synchronize(self):
        type(self).synchronizations += 1

    def elapsed_time(self, other):
        return other.timestamp - self.timestamp


class FakeProfiler:
    def __init__(self, **kwargs):
        assert kwargs['record_shapes'] is False and kwargs['with_stack'] is False
        self.starts = self.stops = self.exports = 0

    def start(self):
        self.starts += 1

    def stop(self):
        self.stops += 1

    def export_chrome_trace(self, path):
        assert self.starts == self.stops == 1
        self.exports += 1
        Path(path).write_text('{"traceEvents": []}')


def check_trace_lifecycle(batch, descriptor, temp):
    activities = [P.torch.profiler.ProfilerActivity.CPU, P.torch.profiler.ProfilerActivity.XPU]
    ranges = []

    @contextmanager
    def record_range(name):
        ranges.append(name)
        yield

    with patch.object(P.torch.profiler, 'supported_activities', return_value=activities), \
            patch.object(P.torch.profiler, 'profile', side_effect=FakeProfiler), \
            patch.object(P.torch.profiler, 'record_function', side_effect=record_range):
        profile = P.EventCapture()
        fn = lambda value: value + 1
        assert profile.traced_call('target_body', batch, descriptor, fn, (5,), {}) == 6
        assert not ranges and profile.trace is None
        profile.begin_trace(2)
        fake = profile.trace
        try:
            profile.begin_trace(2)
        except AssertionError:
            pass
        else:
            raise AssertionError('Overlapping trace was accepted')
        batch.has_prefill = True
        assert profile.traced_call('target_body', batch, descriptor, fn, (5,), {}) == 6
        profile.trace_step_complete()
        assert fake.starts == 0 and not ranges and profile.trace_cycles == 0
        try:
            profile.finish_trace(Path(temp) / 'unfinished.json')
        except AssertionError:
            pass
        else:
            raise AssertionError('Unfinished trace was accepted')
        batch.has_prefill = False

        assert profile.traced_call('target_body', batch, descriptor, fn, (5,), {}) == 6
        assert profile.traced_call('target_head', batch, None, fn, (6,), {}) == 7
        profile.trace_step_complete()
        assert fake.starts == 1 and fake.stops == 0
        # A later mixed cycle must be labeled honestly, not called pure decode.
        batch.has_prefill = True
        assert profile.traced_call('target_body', batch, descriptor, fn, (5,), {}) == 6
        profile.trace_step_complete()
        assert profile.trace_complete and fake.stops == 1
        before = len(ranges)
        profile.traced_call('target_body', batch, descriptor, fn, (5,), {})
        profile.trace_step_complete()
        assert len(ranges) == before and fake.stops == 1
        path = Path(temp) / 'bounded-trace.json'
        profile.finish_trace(path)
        meta = json.loads(Path(str(path) + '.meta.json').read_text())
        assert meta['runner_cycles'] == 2 and meta['pure_decode_cycles'] == meta['mixed_or_prefill_cycles'] == 1
        assert ranges == ['exl3_diagnostic/cycle0/target_body', 'exl3_diagnostic/cycle0/target_head',
                          'exl3_diagnostic/cycle1/target_body']
        assert fake.exports == 1 and profile.trace is None
        with patch.object(P.torch.profiler, 'supported_activities', return_value=[]):
            try:
                profile.begin_trace(2)
            except AssertionError:
                pass
            else:
                raise AssertionError('Missing XPU profiler support was accepted')
        batch.has_prefill = False


def check_vocabulary_inventory():
    extension = P.ProfileWorkerExtension()
    heads = [SimpleNamespace(exl3_loader_report={'members': ['lm_head'], 'bits': 6},
                             svh=P.torch.empty(248320)) for _ in range(2)]
    models = [SimpleNamespace(named_modules=lambda h=h: [('lm_head', h)]) for h in heads]
    extension.model_runner = SimpleNamespace(model=models[0], speculator=SimpleNamespace(model=models[1]))
    with patch.object(P.torch.xpu, 'mem_get_info', return_value=(777, 888)), \
            patch.object(P.torch.xpu, 'memory_allocated', return_value=123), \
            patch.object(P.torch.xpu, 'memory_reserved', return_value=456):
        full = extension.vocabulary_inventory()
        assert full['torch_allocated_bytes'] == 123 and full['device_free_bytes'] == 777
        assert all(h['full_rows'] == 248320 and h['pruned_rows'] is None for h in full['heads'])
        for head in heads:
            head.exl3_draft = {'svh': P.torch.empty(65536), 'idx': P.torch.empty(65536, dtype=P.torch.int64),
                               'bounds': [0, 65536]}
        pruned = extension.vocabulary_inventory()
        assert all(h['pruned_rows'] == 65536 and h['pruned_tensor_logical_bytes'] == 65536 * 12
                   for h in pruned['heads'])


class Model:
    def compute_logits(self, hidden):
        FakeEvent.clock += 10
        return hidden + 1

    def compute_logits_local(self, hidden):
        FakeEvent.clock += 20
        return hidden + 2


class Runner:
    def __init__(self, sharded=False, rejected=False):
        self.model = Model()
        self.sharded = sharded
        self.rejected = rejected

    def sampler(self, logits, batch):
        FakeEvent.clock += 3
        return logits * 2

    def rejection_sampler(self, logits, batch, draft):
        FakeEvent.clock += 7
        return logits * 3

    def sample(self, hidden, input_batch, grammar_output):
        if self.sharded:
            local_logits = self.model.compute_logits_local(hidden)
            logits = local_logits
        else:
            logits = self.model.compute_logits(hidden)
        if self.rejected:
            output = self.rejection_sampler(logits, input_batch, None)
        else:
            output = self.sampler(logits, input_batch)
        return output


class ChangedRunner:
    def sample(self, hidden, input_batch, grammar_output):
        return hidden


def main():
    real = object.__new__(GPUModelRunner)
    identities = {name: P.patch_method(getattr(real, name), name)[1] for name in P.CALL_SITES}
    try:
        P.patch_method(ChangedRunner().sample, "sample")
    except AssertionError:
        pass
    else:
        raise AssertionError("Changed V2 call sites were accepted")
    batch = SimpleNamespace(req_ids=["a", "b", "c", "d"], num_reqs=4,
        num_tokens=20, num_tokens_after_padding=24, num_scheduled_tokens=np.array([5] * 4),
        num_computed_prefill_tokens_np=np.array([49152] * 4), prefill_len_np=np.array([49152] * 4),
        has_prefill=False, num_draft_tokens=16)
    descriptor = SimpleNamespace(cg_mode="FULL", num_tokens=24)
    assert P.batch_metadata(batch, descriptor)["actual_rows"] == 20
    assert P.batch_metadata(batch, descriptor)["target_graph_bucket_rows"] == 24
    with tempfile.TemporaryDirectory() as temp, patch.object(P.torch.xpu, "Event", FakeEvent), \
            patch.object(P.torch.xpu, "current_stream", return_value="fake-stream"):
        for sharded, rejected in [(False, False), (False, True), (True, False), (True, True)]:
            r = Runner(sharded, rejected)
            profile = P.EventCapture(); profile.sources = identities
            r._exl3_event_capture = profile
            method, _ = P.patch_method(r.sample, "sample")
            baseline = r.sample(5, batch, None)
            # Disabled capture preserves calls and does not create GPU events.
            assert method(5, batch, None) == baseline and not profile.cycles
            FakeEvent.synchronizations = 0
            profile.enabled = True; profile.begin()
            assert method(5, batch, None) == baseline
            profile.end()
            assert FakeEvent.synchronizations == 0
            out = Path(temp) / f"{sharded}-{rejected}.json"
            profile.finish(out)
            rows = json.loads(out.read_text())["cycles"]
            assert len(rows) == 1 and FakeEvent.synchronizations == 3
            stages = rows[0]["stages"]
            assert [s["stage"] for s in stages] == ["target_head", "verification_sampler"]
            head = 20 if sharded else 10
            sampler = 7 if rejected else 3
            assert [s["gpu_timeline_ms"] for s in stages] == [head, sampler]
            assert rows[0]["cycle_gpu_timeline_ms"] == head + sampler
            assert not profile.enabled and not profile.cycles
        check_trace_lifecycle(batch, descriptor, temp)
        check_vocabulary_inventory()
    print(json.dumps({"status": "PASS", "real_v2_call_sites": identities,
        "changed_call_sites_rejected": True, "timing_passthrough_branches": 4,
        "synchronization_after_generation_only": True,
        "bounded_trace_wait_start_stop_export_and_mixed_metadata": True,
        "full_and_pruned_head_inventory_cpu_check": True,
        "scope": "CPU source/plumbing checks with fake events; no claim of actual XPU timing validation"}))


if __name__ == "__main__":
    main()

"""Check real V2 call-site guards and event plumbing without using a GPU."""
import json
from pathlib import Path
import tempfile
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
    print(json.dumps({"status": "PASS", "real_v2_call_sites": identities,
        "changed_call_sites_rejected": True, "timing_passthrough_branches": 4,
        "synchronization_after_generation_only": True,
        "scope": "CPU source/plumbing checks with fake events; no claim of actual XPU timing validation"}))


if __name__ == "__main__":
    main()

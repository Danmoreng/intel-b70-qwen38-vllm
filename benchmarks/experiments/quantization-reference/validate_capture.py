"""Exercise both runner APIs, split prefill, and exclusion of the final logit."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from native_capture import install_capture, consolidate


def logits_fn(h):
    result = torch.zeros(len(h), 248320)
    for i, token in enumerate(h[:, 0].tolist()):
        result[i, 7 if token == 5 else 9] = 3 if token == 5 else 4
    return result


def compute_prompt_logprobs_with_chunking(hidden_states, logits_fn):
    CHUNK_SIZE = 1024
    for start in range(0, len(hidden_states), CHUNK_SIZE):
        logits_fn(hidden_states[start:start + CHUNK_SIZE])


class WorkerV2:
    def compute_prompt_logprobs(self, logits_fn, hidden_states, input_batch,
                                all_token_ids, num_computed_tokens, prompt_lens):
        # Includes the final, unscored input token.
        return compute_prompt_logprobs_with_chunking(hidden_states, logits_fn)


class RequestStatesV2:
    def add_request(self, req_id, prompt_len, all_token_ids, num_computed_tokens, max_tokens):
        self.added = (req_id, prompt_len, list(all_token_ids), num_computed_tokens, max_tokens)
        return "registration_preserved"


class ForbiddenDeviceMetadata:
    def __getitem__(self, index):
        raise AssertionError("Capture must not read device request metadata")


class RunnerV1:
    def __init__(self):
        self.model = SimpleNamespace(compute_logits=logits_fn)

    def _get_prompt_logprobs_dict(self, hidden_states, metadata):
        request, start_idx = metadata
        logits = self.model.compute_logits(hidden_states)
        return logits


def main():
    torch.set_num_threads(2)
    ids = [5, 7, 9]
    expected = -torch.log_softmax(logits_fn(torch.tensor([[5], [7]])), -1)[torch.arange(2), torch.tensor(ids[1:])].numpy()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        panel = root / "panel.json"
        panel.write_text(json.dumps({"windows": [{"ids": ids, "name": "synthetic",
            "domain": "test", "kl_positions": [0, 1], "token_ids_sha256": "fixture"}]}))
        for version in [1, 2]:
            out = root / str(version)
            runner = RunnerV1() if version == 1 else SimpleNamespace(
                prompt_logprobs_worker=WorkerV2(), req_states=RequestStatesV2())
            install_capture(runner, panel, out)
            if version == 1:
                request = SimpleNamespace(prompt_token_ids=ids)
                runner._get_prompt_logprobs_dict(torch.tensor([[5]]), (request, 0))
                runner._get_prompt_logprobs_dict(torch.tensor([[7]]), (request, 1))
            else:
                try:
                    runner.req_states.add_request("bad", 3, [5, 7, 8], 0, 1)
                except AssertionError:
                    assert not hasattr(runner.req_states, "added")
                else:
                    raise AssertionError("Changed prompt was accepted")
                result = runner.req_states.add_request("request", prompt_len=3,
                    all_token_ids=ids, num_computed_tokens=0, max_tokens=1)
                assert result == "registration_preserved"
                assert runner.req_states.added == ("request", 3, ids, 0, 1)
                batch = SimpleNamespace(req_ids=["request"], idx_mapping_np=np.asarray([4]),
                    num_computed_prefill_tokens_np=np.asarray([0]))
                all_ids = ForbiddenDeviceMetadata()
                computed = ForbiddenDeviceMetadata()
                lengths = np.zeros(8, dtype=int); lengths[4] = 3
                runner.prompt_logprobs_worker.compute_prompt_logprobs(logits_fn, torch.tensor([[5]]), batch, all_ids, computed, lengths)
                batch.num_computed_prefill_tokens_np[0] = 1
                runner.prompt_logprobs_worker.compute_prompt_logprobs(logits_fn, torch.tensor([[7], [9]]), batch, all_ids, computed, lengths)
            consolidate(panel, out, 1)
            actual = np.load(out / "window-000-nll.npy")
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            assert np.load(out / "window-000-logprobs.npy").shape == (2, 248320)
            print(json.dumps({"runner_version": version, "split_prefill_alignment": True,
                              "last_logit_excluded": True, "full_vocabulary": 248320,
                              "v2_device_metadata_reads": 0 if version == 2 else None}))
        # Exercise the bounded head chunk plus a partial tail with the same
        # full-vocabulary arithmetic; avoid relying only on a tiny head batch.
        long_ids = [5] + [7] * 258
        panel.write_text(json.dumps({"windows": [{"ids": long_ids, "name": "split-head",
            "domain": "test", "kl_positions": [127, 128, 257], "token_ids_sha256": "head-fixture"}]}))
        runner = SimpleNamespace(prompt_logprobs_worker=WorkerV2(), req_states=RequestStatesV2())
        out = root / "head-split"
        identity = install_capture(runner, panel, out)
        assert identity["measurement_head_chunk_tokens"] == 128
        runner.req_states.add_request("head", len(long_ids), long_ids, 0, 1)
        batch = SimpleNamespace(req_ids=["head"], idx_mapping_np=np.asarray([0]),
            num_computed_prefill_tokens_np=np.asarray([0]))
        runner.prompt_logprobs_worker.compute_prompt_logprobs(logits_fn,
            torch.tensor(long_ids)[:, None], batch, ForbiddenDeviceMetadata(),
            ForbiddenDeviceMetadata(), np.asarray([len(long_ids)]))
        consolidate(panel, out, 1)
        actual = np.load(out / "window-000-nll.npy")
        np.testing.assert_allclose(actual[0], expected[0], atol=1e-6)
        repeated_nll = -torch.log_softmax(logits_fn(torch.tensor([[7]])), -1)[0, 7].item()
        np.testing.assert_allclose(actual[1:], repeated_nll, atol=1e-6)
        chunks = [json.loads(p.read_text()) for p in sorted(out.glob("*chunk-*.json"))]
        assert [(c["start"], c["num_logits"]) for c in chunks] == [(0, 128), (128, 128), (256, 2)]
        # The production chunker is not mutated by the local scorer clone.
        seen = []
        compute_prompt_logprobs_with_chunking(torch.zeros(259, 1), lambda h: seen.append(len(h)))
        assert seen == [259]
        print(json.dumps({"runner_version": 2, "measurement_head_chunks": [128, 128, 2],
                          "original_scorer_unchanged": True, "head_boundary_alignment": True}))


if __name__ == "__main__":
    main()

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


class WorkerV2:
    def compute_prompt_logprobs(self, logits_fn, hidden_states, input_batch,
                                all_token_ids, num_computed_tokens, prompt_lens):
        # Two internal head chunks, including the final, unscored input token.
        for h in hidden_states.split(1):
            logits_fn(h)


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
            runner = RunnerV1() if version == 1 else SimpleNamespace(prompt_logprobs_worker=WorkerV2())
            install_capture(runner, panel, out)
            if version == 1:
                request = SimpleNamespace(prompt_token_ids=ids)
                runner._get_prompt_logprobs_dict(torch.tensor([[5]]), (request, 0))
                runner._get_prompt_logprobs_dict(torch.tensor([[7]]), (request, 1))
            else:
                batch = SimpleNamespace(req_ids=["request"], idx_mapping_np=np.asarray([4]))
                all_ids = torch.zeros(8, 3, dtype=torch.long); all_ids[4] = torch.tensor(ids)
                computed = torch.zeros(8, dtype=torch.long)
                lengths = np.zeros(8, dtype=int); lengths[4] = 3
                runner.prompt_logprobs_worker.compute_prompt_logprobs(logits_fn, torch.tensor([[5]]), batch, all_ids, computed, lengths)
                computed[4] = 1
                runner.prompt_logprobs_worker.compute_prompt_logprobs(logits_fn, torch.tensor([[7], [9]]), batch, all_ids, computed, lengths)
            consolidate(panel, out, 1)
            actual = np.load(out / "window-000-nll.npy")
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            assert np.load(out / "window-000-logprobs.npy").shape == (2, 248320)
            print(json.dumps({"runner_version": version, "split_prefill_alignment": True,
                              "last_logit_excluded": True, "full_vocabulary": 248320}))


if __name__ == "__main__":
    main()

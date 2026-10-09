"""Capture exact full-vocabulary prompt distributions from native vLLM workers.

API prompt_logprobs stays at 1; full distributions never become giant Python
token dictionaries. The hook runs only inside prompt scoring, before sampling.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import types

import numpy as np
import torch


class CaptureWorkerExtension:
    def install_prompt_capture(self, panel_path, output_dir):
        return install_capture(self.model_runner, panel_path, output_dir)


def install_capture(runner, panel_path, output_dir):
    panel = json.loads(Path(panel_path).read_text())
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lookup = {tuple(w["ids"]): (i, w) for i, w in enumerate(panel["windows"])}

    def capture(logits, ids, start):
        key = tuple(ids)
        assert key in lookup, "Prompt token IDs differ from frozen panel"
        i, window = lookup[key]
        start = int(start)
        n = logits.shape[0]
        assert logits.shape[1] == 248320, f"Unexpected vocabulary size {logits.shape}"
        positions, distributions, nlls = [], [], []
        for offset in range(0, n, 64):
            end = min(n, offset + 64)
            lp = torch.log_softmax(logits[offset:end].float(), dim=-1)
            assert torch.isfinite(lp).all(), "Nonfinite native log probabilities"
            targets = torch.tensor(ids[start + offset + 1:start + end + 1], device=lp.device)
            assert len(targets) == end - offset
            nlls.extend((-lp.gather(1, targets[:, None]).squeeze(1)).cpu().tolist())
            for p in window["kl_positions"]:
                if start + offset <= p < start + end:
                    positions.append(p)
                    distributions.append(lp[p - start - offset].cpu().numpy())
        stem = out / f"window-{i:03d}-chunk-{start:05d}"
        np.save(str(stem) + "-nll.npy", np.asarray(nlls, dtype=np.float32))
        if distributions:
            np.save(str(stem) + "-logprobs.npy", np.stack(distributions))
        Path(str(stem) + ".json").write_text(json.dumps({"window": i, "start": start,
            "num_logits": n, "kl_positions": positions,
            "token_ids_sha256": window["token_ids_sha256"]}) + "\n")
        return logits

    if hasattr(runner, "prompt_logprobs_worker"):
        worker = runner.prompt_logprobs_worker
        assert worker is not None
        original_v2 = worker.compute_prompt_logprobs
        # The normal serving profile does not materialize prompt distributions.
        # Its .965 KV reservation leaves too little headroom for the scorer's
        # default 1024 x 248320 logits plus FP32 top-k/normalization temporaries.
        # Bound only this measurement's head batches, keeping the scorer and
        # its target IDs/normalization otherwise unchanged.
        head_chunk_tokens = 128
        original_chunker = original_v2.__func__.__globals__["compute_prompt_logprobs_with_chunking"]
        chunk_source = textwrap.dedent(inspect.getsource(original_chunker))
        scorer_file = Path(inspect.getfile(original_chunker))
        assert hashlib.sha256(scorer_file.read_bytes()).hexdigest() == \
            "d65ae509ee6da4f4bedd7626807462a91bf6817613d907b02880c18b9af45d27", \
            "vLLM 0.31 prompt scorer source changed"
        assert original_chunker.__globals__["CHUNK_SIZE"] == 1024
        # vLLM 0.31 moved the same chunk constant into the module globals.
        # Use a private globals copy; production scoring remains untouched.
        chunk_namespace = dict(original_chunker.__globals__)
        chunk_namespace["CHUNK_SIZE"] = head_chunk_tokens
        chunk_namespace[original_chunker.__name__] = types.FunctionType(
            original_chunker.__code__, chunk_namespace, original_chunker.__name__,
            original_chunker.__defaults__, original_chunker.__closure__)
        scorer_namespace = dict(original_v2.__func__.__globals__)
        scorer_namespace["compute_prompt_logprobs_with_chunking"] = chunk_namespace[original_chunker.__name__]
        bounded_v2 = types.MethodType(types.FunctionType(original_v2.__func__.__code__,
            scorer_namespace, original_v2.__name__, original_v2.__func__.__defaults__,
            original_v2.__func__.__closure__), worker)
        # V2 keeps request token IDs on the device, but receives the same IDs
        # on the CPU at registration. Keep that immutable prompt copy instead
        # of introducing a blocking device read into async prompt scoring.
        original_add = runner.req_states.add_request
        add_signature = inspect.signature(original_add)
        assert {"req_id", "prompt_len", "all_token_ids"} <= set(add_signature.parameters), \
            "V2 request registration changed; refusing ambiguous instrumentation"
        registered_prompts = {}

        def add_request(*args, **kwargs):
            bound = add_signature.bind(*args, **kwargs)
            length = int(bound.arguments["prompt_len"])
            ids = tuple(bound.arguments["all_token_ids"][:length])
            assert len(ids) == length and ids in lookup, \
                "Registered prompt token IDs differ from frozen panel"
            result = original_add(*args, **kwargs)
            registered_prompts[bound.arguments["req_id"]] = ids
            return result

        runner.req_states.add_request = add_request

        def compute_prompt_logprobs(logits_fn, hidden_states, input_batch,
                                    all_token_ids, num_computed_tokens, prompt_lens):
            assert len(input_batch.req_ids) == 1, "Capture requires one request per batch"
            state = int(input_batch.idx_mapping_np[0])
            length = int(prompt_lens[state])
            ids = registered_prompts[input_batch.req_ids[0]]
            assert len(ids) == length, "Registered prompt length changed"
            # This is the batch's pre-forward CPU prefill offset. The request
            # state's optimistic computed-token count may already be ahead.
            start = int(input_batch.num_computed_prefill_tokens_np[0])
            assert start >= 0
            cursor = 0

            def captured_logits(h):
                nonlocal cursor
                logits = logits_fn(h)
                valid = min(len(logits), length - 1 - start - cursor)
                if valid > 0:
                    capture(logits[:valid], ids, start + cursor)
                cursor += len(logits)
                return logits

            result = bounded_v2(captured_logits, hidden_states, input_batch,
                               all_token_ids, num_computed_tokens, prompt_lens)
            if hasattr(torch, "xpu") and torch.xpu.is_initialized():
                # Allocator counters are host metadata, not a device copy/sync.
                (out / "capture-memory.json").write_text(json.dumps({
                    "prompt_start": start, "prompt_length": length,
                    "allocated_bytes": torch.xpu.memory_allocated(),
                    "reserved_bytes": torch.xpu.memory_reserved(),
                    "peak_allocated_bytes": torch.xpu.max_memory_allocated(),
                    "peak_reserved_bytes": torch.xpu.max_memory_reserved(),
                    "head_chunk_tokens": head_chunk_tokens}) + "\n")
            return result

        worker.compute_prompt_logprobs = compute_prompt_logprobs
        source = inspect.getsource(original_v2)
        return {"capture_installed": True, "runner_type": type(runner).__name__,
                "prompt_scorer_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "request_registration_sha256": hashlib.sha256(inspect.getsource(original_add).encode()).hexdigest(),
                "original_head_chunker_sha256": hashlib.sha256(chunk_source.encode()).hexdigest(),
                "measurement_head_chunk_tokens": head_chunk_tokens,
                "prompt_metadata_source": "registered_cpu_ids_and_batch_cpu_prefill_offset",
                "panel_sha256": hashlib.sha256(Path(panel_path).read_bytes()).hexdigest()}

    original = runner._get_prompt_logprobs_dict.__func__
    # Preserve this version's prompt scorer verbatim, inserting just a
    # passthrough capture immediately after its actual compute_logits call.
    # Patching an nn.Module instance's compute_logits is insufficient on XPU:
    # the runtime can use a different wrapped model for prompt scoring.
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    matches = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "logits":
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "compute_logits":
                node.value = ast.Call(func=ast.Name(id="_full_vocab_capture", ctx=ast.Load()),
                    args=[value, ast.Attribute(value=ast.Name(id="request", ctx=ast.Load()), attr="prompt_token_ids", ctx=ast.Load()),
                          ast.Name(id="start_idx", ctx=ast.Load())], keywords=[])
                matches += 1
    assert matches == 1, "vLLM prompt scorer changed; refusing ambiguous instrumentation"
    namespace = dict(original.__globals__)
    namespace["_full_vocab_capture"] = capture
    exec(compile(ast.fix_missing_locations(tree), original.__code__.co_filename + ":full-vocab-capture", "exec"), namespace)
    runner._get_prompt_logprobs_dict = types.MethodType(namespace[original.__name__], runner)
    return {"capture_installed": True, "runner_type": type(runner).__name__,
            "prompt_scorer_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "panel_sha256": hashlib.sha256(Path(panel_path).read_bytes()).hexdigest()}


def consolidate(panel_path, output_dir, windows):
    panel = json.loads(Path(panel_path).read_text())
    out = Path(output_dir)
    summaries = []
    for i, w in enumerate(panel["windows"][:windows]):
        nll = np.full(len(w["ids"]) - 1, np.nan, dtype=np.float32)
        selected, visited = {}, set()
        for f in sorted(out.glob(f"window-{i:03d}-chunk-*.json")):
            meta = json.loads(f.read_text())
            assert meta["token_ids_sha256"] == w["token_ids_sha256"]
            start, count = meta["start"], meta["num_logits"]
            indices = set(range(start, start + count))
            assert not visited.intersection(indices), "Duplicate/recomputed prompt chunk"
            visited.update(indices)
            nll[start:start + count] = np.load(str(f.with_suffix("")) + "-nll.npy")
            if meta["kl_positions"]:
                rows = np.load(str(f.with_suffix("")) + "-logprobs.npy")
                selected.update(zip(meta["kl_positions"], rows))
        assert np.isfinite(nll).all(), "Missing prompt NLL positions"
        assert set(selected) == set(w["kl_positions"]), "Missing KL positions"
        np.save(out / f"window-{i:03d}-nll.npy", nll)
        np.save(out / f"window-{i:03d}-logprobs.npy", np.stack([selected[p] for p in w["kl_positions"]]))
        summaries.append({"window": i, "name": w["name"], "domain": w["domain"],
                          "positions": len(nll), "nll_mean": float(nll.mean()),
                          "perplexity": float(np.exp(nll.mean()))})
    return summaries

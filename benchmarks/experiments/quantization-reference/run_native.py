import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from vllm import LLM, SamplingParams
from native_capture import consolidate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--quantization", choices=["gptq", "exl3"], required=True)
    p.add_argument("--panel", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--windows", type=int, default=0)
    p.add_argument("--engine-config", help="JSON engine overrides for a named precision stage")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError("Fresh native capture directory required")
    panel = json.loads(Path(args.panel).read_text())
    windows = panel["windows"][:args.windows] if args.windows else panel["windows"]
    started = time.monotonic()
    engine = dict(dtype="float16", max_model_len=2048, max_num_seqs=1,
                  max_num_batched_tokens=1024, gpu_memory_utilization=0.80,
                  enforce_eager=True, enable_prefix_caching=False, max_logprobs=1,
                  limit_mm_per_prompt={"image": 0, "video": 0}, seed=20261001)
    if args.engine_config:
        overrides = json.loads(Path(args.engine_config).read_text())
        if {"model", "quantization", "worker_extension_cls"}.intersection(overrides):
            raise ValueError("Engine stage may not replace model/quantization/capture")
        engine.update(overrides)
    print(json.dumps({"engine_config": engine}), flush=True)
    llm = LLM(model=args.model, quantization=args.quantization, **engine,
              worker_extension_cls="native_capture.CaptureWorkerExtension")
    capture_identity = llm.collective_rpc("install_prompt_capture", args=(args.panel, args.out))
    print(capture_identity, flush=True)
    sp = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
    outputs = []
    for i, w in enumerate(windows):
        result = llm.generate([{"prompt_token_ids": w["ids"]}], sp, use_tqdm=False)[0]
        assert result.prompt_token_ids == w["ids"]
        assert list(out.glob(f"window-{i:03d}-chunk-*.json")), "Native prompt capture did not execute"
        native_nll = [-row[token].logprob for row, token in zip(result.prompt_logprobs[1:], w["ids"][1:])]
        outputs.append(np.asarray(native_nll, dtype=np.float32))
        print(json.dumps({"completed_window": i, "name": w["name"],
                          "api_nll_mean": float(np.mean(native_nll))}), flush=True)
    summaries = consolidate(args.panel, args.out, len(windows))
    # Cross-check capture alignment against the independent prompt-logprobs API.
    for i, api_nll in enumerate(outputs):
        captured = np.load(out / f"window-{i:03d}-nll.npy")
        assert np.allclose(captured, api_nll, atol=2e-5, rtol=2e-6), "Capture/API alignment failed"
    nll = np.concatenate(outputs)
    summary = {"kind": "native_vllm_teacher_forced", "model_path": args.model,
               "quantization": args.quantization, "dtype": "float16",
               "panel_sha256": hashlib.sha256(Path(args.panel).read_bytes()).hexdigest(),
               "vocab_size": 248320, "windows": summaries, "positions": len(nll),
               "nll_mean": float(nll.mean()), "perplexity": float(np.exp(nll.mean())),
               "seconds": time.monotonic() - started, "torch": torch.__version__,
               "api_alignment_verified": True, "engine_config": engine,
               "capture_identity": capture_identity,
               "kv_cache_dtype": engine.get("kv_cache_dtype", "auto/float16"),
               "speculation": bool(engine.get("speculative_config")),
               "prefix_cache": engine["enable_prefix_caching"],
               "graphs": not engine["enforce_eager"],
               "scope": "Teacher-forced prompt scoring. Configuring graphs/MTP does not by itself prove decode graph or draft acceptance correctness."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

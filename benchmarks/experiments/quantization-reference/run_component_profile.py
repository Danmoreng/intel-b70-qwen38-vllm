"""Profile actual decode rows and XPU-event spans on a frozen input panel."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096, 49152, 102752])
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    assert not args.out.exists(), "Fresh profile evidence required"
    args.out.mkdir()
    raw = args.panel.read_bytes()
    raw = gzip.decompress(raw) if args.panel.suffix == ".gz" else raw
    panel = json.loads(raw)
    windows = [w for w in panel["windows"] if w["context_tokens"] in args.contexts]
    assert len(windows) == 2 * len(args.contexts)
    cases = [(w, c) for w in windows for c in [1, 4]]
    if args.compact:
        assert args.output_tokens == 128
        cases = [(next(w for w in panel['windows'] if w['name'] == name), c)
                 for name, c in [('code-4096', 1), ('code-49152', 4), ('code-102752', 1)]]
    config = json.loads(args.engine_config.read_text())
    assert not {"model", "quantization", "worker_extension_cls"}.intersection(config)
    config.update(dtype="float16", seed=20261001)
    llm = LLM(model=args.model, quantization="exl3", **config,
              worker_extension_cls="profile_capture.ProfileWorkerExtension")
    sp = SamplingParams(temperature=0, top_p=1, top_k=-1, seed=20261001,
                        max_tokens=args.output_tokens, ignore_eos=True)
    short = next(w for w in panel["windows"] if w["name"] == "code-4096")
    for concurrency in [1, 4]:
        llm.generate([{"prompt_token_ids": short["ids"], "cache_salt": f"profile-warmup-{concurrency}-{i}"}
                      for i in range(concurrency)],
                     SamplingParams(temperature=0, max_tokens=32, ignore_eos=True), use_tqdm=False)
    inventory = llm.collective_rpc("vocabulary_inventory")
    (args.out / 'vocabulary-inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    identity = llm.collective_rpc("install_event_profile")
    rows, traces = [], []
    for window, concurrency in cases:
        name = f'{window["name"]}-c{concurrency}'
        llm.collective_rpc("begin_event_profile")
        start = time.monotonic()
        results = llm.generate([{"prompt_token_ids": window["ids"], "cache_salt": f"profile-{name}-{i}"}
                                for i in range(concurrency)], sp, use_tqdm=False)
        elapsed = time.monotonic() - start
        assert len(results) == concurrency
        for result in results:
            assert result.prompt_token_ids == window["ids"]
            assert len(result.outputs) == 1 and len(result.outputs[0].token_ids) == args.output_tokens
        capture = llm.collective_rpc("finish_event_profile", args=(str(args.out / (name + ".json")),))
        outputs = [result.outputs[0].token_ids for result in results]
        row = {"name": name, "context_tokens": len(window["ids"]), "concurrency": concurrency,
               "generated_tokens": concurrency * args.output_tokens, "diagnostic_wall_s": elapsed,
               "output_ids_sha256": hashlib.sha256(json.dumps(outputs, separators=(",", ":")).encode()).hexdigest(),
               "capture": capture}
        rows.append(row)
        (args.out / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if (len(window['ids']), concurrency) not in [(102752, 1), (49152, 4)]:
            continue
        trace_name = f'kernel-trace-{name}'
        llm.collective_rpc("begin_kernel_trace", args=(8,))
        # Reuse the immediately preceding prompt/salt. Report actual hits;
        # never assume the entire prefix remained cached.
        results = llm.generate([{"prompt_token_ids": window["ids"], "cache_salt": f"profile-{name}-{i}"}
                                for i in range(concurrency)],
                               SamplingParams(temperature=0, top_p=1, top_k=-1, seed=20261001,
                                              max_tokens=128, ignore_eos=True), use_tqdm=False)
        assert len(results) == concurrency
        assert all(result.prompt_token_ids == window["ids"] and len(result.outputs[0].token_ids) == 128
                   for result in results)
        capture = llm.collective_rpc("finish_kernel_trace", args=(str(args.out / (trace_name + ".json")),))
        traces.append({"name": trace_name, "concurrency": concurrency, "context_tokens": len(window['ids']),
                       "output_tokens_per_request": 128, "capture": capture,
                       "observed_cached_tokens": [getattr(r, 'num_cached_tokens', None) for r in results],
                       "cache_mode": "same prompt and salt immediately after event wave; report actual observed hits"})
        print(json.dumps(traces[-1]), flush=True)
    summary = {"status": "COMPLETE", "kind": "diagnostic_xpu_event_spans",
               "panel_sha256": hashlib.sha256(raw).hexdigest(), "engine_config": config,
               "capture_identity": identity, "waves": rows, "kernel_traces": traces,
               "compact": args.compact,
               "vocabulary_inventory": inventory,
               "scope": "Instrumented diagnostic; not unbiased throughput. Fused draft graphs are timed as a complete proposal, not decomposed into body/head/sampler."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

# M09: Intel Compute Runtime and IGC

Candidate-only rerun of the frozen production coding workload, requested on
2026-09-19. Promoted to production after review and explicit user approval on
2026-09-19. The repository root README is now the current deployment recipe.

Completed and reviewed: [results from 2026-09-19](RESULTS-2026-09-19.md).
Both tasks passed independent acceptance; wall time fell 2.69% and aggregate
decode rose 4.38%. Comparable-context prefill was essentially unchanged.

The candidate adds Compute Runtime 26.35.39758.10 and IGC 2.41.5 (build 22716)
to the immutable production image. `build.py` pins official package checksums;
`verify-image.py` checks installed package differences and inference code hashes.
No vLLM, PyTorch, model, attention patch, activation dtype, or MTP settings change.

Unchanged: Qwen3.8-27B GPTQ INT4/G128, FP16 target activations/output head,
MTP4 with quantized draft head/linears, Q128/M04, FP8 KV cache, 200704-token
context limit, batch 4096, one sequence, prefix caching, and 180 W.
Compiler caches are isolated and start empty. Startup, qualification, and the
same 8K warmup as the reference are excluded from the measured coding session.

## Local execution

These scripts use this machine's recorded baseline and sandbox paths.

```sh
python3 benchmarks/experiments/m09-intel-runtime/build.py
python3 benchmarks/experiments/m09-intel-runtime/verify-image.py
python3 benchmarks/experiments/m09-intel-runtime/prepare.py
bash benchmarks/experiments/m09-intel-runtime/start.sh all
```

The user service `b70-m09-runtime-coding.service` obtains the shared experiment
lock and checks that production is idle. It temporarily stops production, runs
the candidate on loopback port 18087, then restores the original production
service. Both the supervisor's finally block and systemd's ExecStopPost attempt
restoration. The study itself does not replace the production image tag or service definition.
The later, separately authorized promotion is recorded in runs/*/promotion.json.

Before coding: unchanged 196 Ki capacity, nonzero MTP acceptance, correct prefix
reuse, full-context marker recall at 200448 input tokens with a 256-token output
budget, and an isolated agent read-tool/continuation smoke test must pass.

The workload, source, dependency tree, agent 0.85.1, evaluator and prompts are
copied from `m06-target-head-int4/runs/coding-20260919-101848`, control arm only.
Limits remain 3600 seconds, 160 requests and 100000 generated tokens. The sandbox
never edits the live dashboard. Raw prompts and outputs remain local under
ignored `runs/`; `LATEST` locates the current run.

The supervisor records state, native per-request prefill/decode times, prefix
hits, MTP acceptance, context bands, energy, and independent validation. It
writes `comparison.json` after evaluation. A timeout or unsuccessful task is
reported separately from a completed task.

Comparison is against the recorded production arm, not a simultaneous control.
Adaptive agent trajectories can differ, so rates must be read alongside context
bands, cache reuse, task outcome and work performed. A speedup is not assumed.

Official releases:
- https://github.com/intel/compute-runtime/releases/tag/26.35.39758.10
- https://github.com/intel/intel-graphics-compiler/releases/tag/v2.41.5

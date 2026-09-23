# M14: MTP/Mamba prefix boundary screen

`run.py` makes prompts of `nS-1`, `nS`, and `nS+1` tokens, where `S` is the
observed scheduler block size. It sends each prompt cold, resends it exactly,
then extends the same token prefix. The script verifies the token IDs of the
shared prefix, uses greedy output with seed 38, and records cache counters,
computed/cached prompt tokens, TTFT and preemptions per request.

Run on an already-started, idle engine:

```bash
python benchmarks/experiments/m14-prefix-boundary/run.py \
  --block-size 1664 --label vllm-030-prefix-fixed \
  --output benchmarks/runs/2026-09-23-prefix-boundary/v030.json
```

The same label gives byte-identical prompts across versions. Restarting the
engine between arms clears its prefix cache. See the
[2026-09-23 result](../../runs/2026-09-23-prefix-boundary/README.md).

For the v0.30 cache-state correctness spot check, run `correctness.py` on the
same image with prefix caching enabled and disabled, restarting the service
between modes and restoring the production configuration afterwards.

# M04 Shared-KV verification attention

This experiment tests whether the five MTP4 verification rows should share
historical FP8 KV loads instead of using the deployed pseudo-sequence split-K
decode path.

The first challenger deliberately uses the existing native paged causal
Q-block kernel. This isolates the intended data-reuse mechanism before writing
a new kernel: all rows consume one block-table row, with bottom-right causal
masking preserving each row's exact visible history.

The gate covers Q lengths 2, 3 and 5, the 1664-token physical page boundary,
and 8K, 65K, 131K and 196K histories. It requires agreement with both the
deployed path and a short-history FP32 oracle, then at least 3% lower median
device time for every Q5 history before any serving experiment.

Start the isolated, restore-safe background run with:

```bash
scripts/start-shared-kv-verification-study.sh
```

## Qualified result

The Q8 packed-query kernel passed all 90 correctness checks. Relative to the
deployed pseudo-sequence verification path, isolated Q5 device time improved
by 70.74% at 8K, 34.95% at 65K, 28.15% at 131K and 25.16% at 196K.

Frozen-prompt, cold-engine ABBA serving at 65K improved median decode
throughput from 59.355 to 62.898 token/s (+5.97%). The candidate's accepted
speculation rate was 1.61 percentage points lower, but this cost is already
included in the end-to-end throughput result. At the 200704-token production
boundary it improved decode throughput from 33.916 to 34.964 token/s (+3.09%)
while both arms performed exactly five preemptions and 16640 recompute tokens.
Prefill and TTFT remained neutral (-0.09% each).

Evidence:

- `runs/run-20260915-165201/`
- `runs/serving-20260915-165859/`
- `runs/long-20260915-172858/`

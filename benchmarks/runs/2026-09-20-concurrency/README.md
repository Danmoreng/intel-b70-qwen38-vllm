# C1-C4 concurrency and scheduler qualification

Measured on 2026-09-20 on one Intel Arc Pro B70 at 180 W with the published
Q128/M04, MTP4, FP8-KV, 200,704-token profile. Requests used exact rendered
prompt lengths, unique leading markers except in the shared-prefix case, and
streaming OpenAI-compatible chat completions. Native vLLM metrics were sampled
every 250 ms to record running/waiting requests, KV usage and preemptions.

## Decision

Promote `MAX_NUM_SEQS=4`, `MAX_NUM_BATCHED_TOKENS=6656`,
`SCHEDULER_WATERMARK=0.0`, with `--scheduler-reserve-full-isl`. The maximum
per-request model context remains 200,704 tokens. C4 is an admission ceiling;
vLLM queues excess requests when combined KV demand is too high.

## C1-C4 throughput at batch 4096 / watermark 0.0

| Workload | C1 | C2 | C3 | C4 | C4 vs C1 |
|---|---:|---:|---:|---:|---:|
| 2K prompt / 512 output | 59.59 | 86.61 | 102.10 | 116.11 | +94.9% |
| 4K coding / 1K output | 64.80 | 93.15 | 110.32 | 129.68 | +100.1% |
| 16K prompt / 512 output | 25.64 | 29.65 | 31.16 | 32.46 | +26.6% |

Values are aggregate output tokens/s. These cases completed with zero
preemptions. Higher concurrency increased aggregate throughput while increasing
per-request TTFT.

## Watermark comparison at batch 4096

| Watermark | 96K/C4 tok/s | 96K preemptions | 48K-growth/C4 tok/s | Growth preemptions |
|---:|---:|---:|---:|---:|
| 0% | 1.968 | 5 | 62.12 | 4 |
| 5% | 1.974 | 4 | 64.40 | 4 |
| 10% | 1.973 | 4 | 57.15 | 4 |

The watermarks changed admission timing but did not remove growth preemptions
or materially improve long admission. Watermark 0.10 reduced growth throughput,
so 0.0 was retained.

## Batch 6656 at C4 / watermark 0.0

| Case | Batch 4096 | Batch 6656 | Result |
|---|---:|---:|---|
| 4K coding / 1K output | 129.68 tok/s | 128.80 tok/s | -0.7%, effectively flat |
| 96K/C4 admission | 1.968 tok/s, 5 preemptions | 2.072 tok/s, 0 preemptions | +5.3%, no recompute |
| KV capacity | 215,143 tokens | 212,255 tokens | -1.34% |

The 6,656-token budget fits four 1,664-token aligned Mamba pages; 4,096 fits
only two complete aligned pages per scheduler step. The larger budget improved
long-prefill progress without materially changing short C4 throughput.

## Functional and correctness checks

- Vision analysis, parsed tool calls and tool-result follow-up passed.
- The post-promotion C4 correctness run completed at 64.46 tok/s with zero
  preemptions and all four requests successful.
- Q1 and Q4 were byte-identical between sequential and concurrent execution.
  Q2 and Q3 selected different late-output variants; repeated sequential
  baselines also changed variants. This is consistent with numerical and
  scheduling nondeterminism, not request mixing.

The public record intentionally contains no prompt or generated-response text.
Aggregate results and the exact serving profile are in `summary.json` and
`profile.json`.

# C1-C4 concurrency and scheduler qualification

Measured on 2026-09-20 on one Intel Arc Pro B70 at 180 W with the published
Q128/M04, MTP4, FP8-KV, 200,704-token profile. Requests used exact rendered
prompt lengths, unique leading markers except in the shared-prefix case, and
streaming OpenAI-compatible chat completions. Native vLLM metrics were sampled
every 250 ms to record running/waiting requests, KV usage and preemptions.
Separate prefill/decode phase-counter deltas were not retained for this run.

## Decision

Promote `MAX_NUM_SEQS=4`, `MAX_NUM_BATCHED_TOKENS=6656`,
`SCHEDULER_WATERMARK=0.0`, with `--scheduler-reserve-full-isl`. The maximum
per-request model context remains 200,704 tokens. C4 is an admission ceiling;
vLLM queues excess requests when combined KV demand is too high.

## C1-C4 completion time at batch 4096 / watermark 0.0

| Workload per request | C1 | C2 batch | C3 batch | C4 batch | Four serial C1 requests | C4 wall-time reduction |
|---|---:|---:|---:|---:|---:|---:|
| 2K prompt / 512 output | 8.59 s | 11.82 s | 15.04 s | 17.64 s | 34.37 s | 48.7% |
| 4K coding / 1K output | 15.80 s | 21.99 s | 27.85 s | 31.59 s | 63.21 s | 50.0% |
| 16K prompt / 512 output | 19.97 s | 34.54 s | 49.30 s | 63.10 s | 79.87 s | 21.0% |

Times cover the complete request wave, including prompt processing and
generation. For example, the C4 value is the wall time until all four requests
finish, not mean latency and not decode time. These cases completed with zero
preemptions. Higher concurrency reduced total wall time versus serial execution
while increasing per-request TTFT.

Prefill and decode throughput must be measured from their matching native phase
counters. Because those deltas are absent here, this record deliberately does
not derive either rate from end-to-end time. The repository's phase-separated
single-request and coding results are published in
[`benchmarks/RESULTS.md`](../../RESULTS.md).

## Watermark comparison at batch 4096

| Watermark | 96K/C4 batch time | 96K preemptions | Growth preemptions |
|---:|---:|---:|---:|
| 0% | 520.43 s | 5 | 4 |
| 5% | 518.80 s | 4 | 4 |
| 10% | 519.09 s | 4 | 4 |

The watermarks changed admission timing but did not remove growth preemptions
or materially improve long-admission completion time. The legacy aggregate
measurement also regressed for the 48K growth case at watermark 0.10, but its
phase split was not retained. Watermark 0.0 was therefore kept.

## Batch 6656 at C4 / watermark 0.0

| Case | Batch 4096 | Batch 6656 | Result |
|---|---:|---:|---|
| 4K coding / 1K output, C4 batch time | 31.59 s | 31.80 s | +0.7%, effectively flat |
| 96K/C4 admission batch time | 520.43 s, 5 preemptions | 494.18 s, 0 preemptions | -5.0%, no recompute |
| KV capacity | 215,143 tokens | 212,255 tokens | -1.34% |

The 6,656-token budget fits four 1,664-token aligned Mamba pages; 4,096 fits
only two complete aligned pages per scheduler step. The larger budget improved
long-prefill progress without materially changing short C4 completion time.

## Functional and correctness checks

- Vision analysis, parsed tool calls and tool-result follow-up passed.
- The post-promotion C4 correctness run completed with zero preemptions and all
  four requests successful. Its legacy aggregate output rate is retained only
  in `summary.json`; phase-separated rates were not captured.
- Q1 and Q4 were byte-identical between sequential and concurrent execution.
  Q2 and Q3 selected different late-output variants; repeated sequential
  baselines also changed variants. This is consistent with numerical and
  scheduling nondeterminism, not request mixing.

The public record intentionally contains no prompt or generated-response text.
The original aggregate output rates are retained in `summary.json` so the
elapsed-time conversion remains auditable. They are legacy source measurements,
not decode-throughput claims. The exact serving profile is in `profile.json`.

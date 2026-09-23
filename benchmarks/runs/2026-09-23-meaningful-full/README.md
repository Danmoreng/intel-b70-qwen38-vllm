# Full source-review serving benchmark — 2026-09-23

Run `run-20260923-201101-w0.00` measured the production vLLM 0.30.0+xpu image with XPU kernels 0.1.15.4 on one Arc Pro B70 at a verified 180 W power cap. Q128/KV32 prefill, M04 shared-KV verification, MTP4, FP8 KV, C4 admission and prefix caching were enabled. Image ID, source hashes and content-free results are in [summary.json](summary.json).

The run took 75.7 minutes. All 20 scenarios, 70 measured waves and 124 requests completed. Every endpoint prompt count matched local tokenization, and every request generated exactly 1,024 completion tokens with `finish_reason=length`. Total preemptions: 0; excess prefill recomputation: 0 tokens.

## Workload and rates

The [default plan](../../current-profile-scenarios.json) covers eight C1 input budgets, C2–C4 at 2K/4K/16K, exact 16K/64K prefix resends and the maximum-context point. Prompts use complete code and documentation files from the [frozen public corpus](../../meaningful-corpus.json), with concrete source-review tasks. Budgets are upper bounds; actual token ranges are shown below. Sampling was temperature 1.0, top-p 0.95, top-k 20 with recorded seeds. Thinking was disabled; `ignore_eos=true` forces exactly 1,024 generated tokens.

Native prefill rate is newly computed KV tokens divided by native prefill seconds. Native decode rate is post-first generated tokens divided by native decode seconds. MTP acceptance is accepted draft tokens divided by drafted tokens. Rates and latencies are per-scenario medians across waves unless indicated otherwise; acceptance is weighted across drafted tokens.

## Cold C1 context sweep

| Input budget / output | Waves | Actual input | Prefill tok/s | Decode tok/s | MTP accepted | TTFT | TPOT | End to end |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 / 1,024 | 5 | 479–507 | 1,739.2 | 71.8 | 46.1% | 0.29 s | 13.92 ms | 14.54 s |
| 2,048 / 1,024 | 5 | 1,992–2,047 | 1,541.6 | 68.4 | 49.0% | 1.30 s | 14.61 ms | 16.29 s |
| 4,096 / 1,024 | 5 | 4,052–4,094 | 1,480.9 | 67.6 | 47.5% | 2.76 s | 14.79 ms | 17.94 s |
| 8,192 / 1,024 | 5 | 8,167–8,186 | 1,414.7 | 64.1 | 45.6% | 5.80 s | 15.60 ms | 21.80 s |
| 16,384 / 1,024 | 5 | 16,335–16,379 | 1,311.0 | 65.7 | 50.3% | 12.51 s | 15.21 ms | 28.06 s |
| 32,768 / 1,024 | 5 | 32,704–32,762 | 1,157.3 | 59.6 | 49.1% | 28.33 s | 16.78 ms | 45.49 s |
| 65,536 / 1,024 | 3 | 65,491–65,532 | 939.9 | 54.7 | 48.7% | 69.79 s | 18.26 ms | 88.48 s |
| 131,072 / 1,024 | 3 | 131,034–131,070 | 680.4 | 46.0 | 52.8% | 192.80 s | 21.71 ms | 215.01 s |

## Concurrent source-review requests

The aggregate rate counts native generated tokens only over sampled intervals after all requests emitted a first token and before any finished. The counter was sampled every 250 ms. C1 values come from the phase sweep. Other load levels contain three waves each. Tasks vary between points, so these are serving-load points rather than paired scaling measurements.

| Input budget / output per request | Load | Actual input | Fully overlapped decode tok/s | Median batch end to end | MTP accepted | Max waiting |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 / 1,024 | C1 | 1,992–2,047 | 69.9 | 16.29 s | 49.0% | 0 |
| 2,048 / 1,024 | C2 | 1,994–2,048 | 119.1 | 20.98 s | 50.0% | 0 |
| 2,048 / 1,024 | C3 | 2,000–2,045 | 159.2 | 23.64 s | 49.3% | 0 |
| 2,048 / 1,024 | C4 | 1,998–2,047 | 204.7 | 26.60 s | 50.0% | 0 |
| 4,096 / 1,024 | C1 | 4,052–4,094 | 67.8 | 17.94 s | 47.5% | 0 |
| 4,096 / 1,024 | C2 | 4,031–4,096 | 115.3 | 23.64 s | 48.5% | 0 |
| 4,096 / 1,024 | C3 | 4,043–4,091 | 155.2 | 29.49 s | 49.5% | 0 |
| 4,096 / 1,024 | C4 | 4,034–4,096 | 189.8 | 33.41 s | 47.3% | 1 |
| 16,384 / 1,024 | C1 | 16,335–16,379 | 66.4 | 28.06 s | 50.3% | 0 |
| 16,384 / 1,024 | C2 | 16,315–16,377 | 101.0 | 45.82 s | 47.7% | 1 |
| 16,384 / 1,024 | C3 | 16,338–16,383 | 127.2 | 63.04 s | 47.3% | 2 |
| 16,384 / 1,024 | C4 | 16,319–16,383 | 165.9 | 76.70 s | 51.6% | 2 |

## Exact prefix resend

Each size used one cold request followed by two exact prompt resends. The 16K rows come from the full suite. Its first 64K request reused 13,312 tokens from the preceding 16K case; the 64K rows below therefore come from a supplementary isolated run with a scenario-specific prefix namespace. Cache and computed token counts are native counter deltas; warm rows give medians of two resends.

| Input budget | State | Actual input | Cached / computed prompt tokens | TTFT | End to end |
|---:|---|---:|---:|---:|---:|
| 16,384 / 1,024 | Cold | 16,382 | 0 / 16,382 | 12.51 s | 26.74 s |
| 16,384 / 1,024 | Warm | 16,382 | 13,312 / 3,070 | 2.51 s | 16.89 s |
| 65,536 / 1,024 | Cold | 65,476 | 0 / 65,476 | 69.73 s | 88.65 s |
| 65,536 / 1,024 | Warm | 65,476 | 63,232 / 2,244 | 3.36 s | 22.92 s |

## Maximum context

The last scenario requests 199,680 input tokens plus 1,024 output tokens, within the configured 200,704-token limit. It is a single capacity-and-throughput observation.

| Actual input / output | Prefill tok/s | Decode tok/s | MTP accepted | TTFT | End to end | Preemptions |
|---:|---:|---:|---:|---:|---:|---:|
| 199,673 / 1,024 | 526.3 | 32.0 | 39.9% | 379.61 s | 411.52 s | 0 |

Raw prompts, response text, streams, metric snapshots and timelines remain in the ignored local directories `benchmark-results/meaningful-full-profile/run-20260923-201101-w0.00/` and `benchmark-results/meaningful-prefix64k-isolated/run-20260923-212648-w0.00/`. The tracked summary contains no prompt or response text.

To rerun the full plan on an exclusive engine:

```bash
python3 scripts/current-profile-benchmark.py --execute
```

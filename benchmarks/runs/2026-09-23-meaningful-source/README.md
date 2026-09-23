# Sampled source-review serving benchmark — 2026-09-23

Run `run-20260923-193431-w0.00` measured the running production vLLM
0.30.0+xpu image with XPU kernels 0.1.15.4 on one Arc Pro B70 at a verified
180 W card cap. Q128/KV32 prefill, M04 shared-KV verification, MTP4, FP8 KV,
C4 admission and prefix caching remained enabled. The image ID, corpus hash,
scenario-plan hash and content-free measurements are in [summary.json](summary.json).
The run took 28 min 05 s including prompt construction and metric collection.

## Workload and checks

The [short plan](../../meaningful-short-scenarios.json) contains eight C1
context points from 512 through 131,072 input-token budgets, C2–C4 waves at
4K, C4 at 16K, and a cold/warm exact resend at 16K. Prompts consist of complete
files from the [frozen public source corpus](../../meaningful-corpus.json),
with concrete review, test or explanation tasks. The input budget is an upper
bound; the table reports the observed token range. Each request used
temperature 1.0, top-p 0.95, top-k 20, a recorded seed, thinking disabled,
`ignore_eos=true` and exactly 1,024 generated tokens.

All 33 waves and 51 requests completed. Endpoint prompt counts matched local
tokenization, every response contained visible text, every completion had
1,024 tokens with `finish_reason=length`, native request accounting matched,
and there were no preemptions or excess prefill recomputation. The eight C1
phase points were cold: their cached-prompt-token deltas were zero. Sample
responses at short, 64K, 128K and C4 contexts were inspected and addressed
the supplied source files. That spot check is not a semantic correctness
score. Forced output can continue beyond a natural answer ending.

Native prefill rate divides newly computed KV tokens by native prefill time.
Native decode rate divides generated tokens after the first by native decode
time. MTP acceptance is accepted draft tokens divided by drafted tokens.
Per-point rates, TTFT, TPOT and end-to-end time below are medians across
waves; MTP acceptance is weighted across draft tokens. Parentheses give the
range of the individual waves.

## Cold C1 context sweep

| Input budget / output | n | Actual input | Prefill tok/s | Decode tok/s | MTP accepted | TTFT | TPOT | End to end |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 / 1,024 | 3 | 495–498 | 1,725.6 | 74.7 (65.5–75.6) | 49.2% (42.9–53.2%) | 0.30 s | 13.39 ms | 13.98 s |
| 2,048 / 1,024 | 3 | 2,029–2,036 | 1,600.7 | 69.4 (61.9–78.1) | 47.8% (40.3–57.1%) | 1.28 s | 14.41 ms | 16.03 s |
| 4,096 / 1,024 | 3 | 4,092–4,096 | 1,501.4 | 72.5 (71.5–76.2) | 53.1% (51.1–56.2%) | 2.74 s | 13.79 ms | 16.85 s |
| 8,192 / 1,024 | 3 | 8,167–8,188 | 1,416.0 | 64.0 (62.7–65.7) | 44.7% (43.3–46.4%) | 5.79 s | 15.64 ms | 21.82 s |
| 16,384 / 1,024 | 3 | 16,342–16,381 | 1,312.8 | 65.6 (64.4–67.2) | 49.3% (47.7–51.0%) | 12.49 s | 15.23 ms | 28.08 s |
| 32,768 / 1,024 | 3 | 32,712–32,763 | 1,158.1 | 62.6 (54.0–66.3) | 48.2% (40.2–55.3%) | 28.30 s | 15.97 ms | 44.64 s |
| 65,536 / 1,024 | 3 | 65,483–65,531 | 940.6 | 54.1 (52.3–54.7) | 48.6% (46.8–50.0%) | 69.74 s | 18.48 ms | 88.64 s |
| 131,072 / 1,024 | 2 | 131,017–131,069 | 681.4 | 42.9 (41.3–44.5) | 47.4% (44.7–50.2%) | 192.48 s | 23.33 ms | 216.35 s |

The 64K requests all stayed near 47–50% MTP acceptance. The earlier 17.3%
point used repeated `x` and greedy decoding on vLLM 0.29; this run does not
repeat that low value. These workloads differ in prompt, sampling, output
length and engine version, so their rates do not form a version A/B comparison.

## Concurrent decode

The aggregate rate counts native generated tokens only during the sampled
interval after all requests emitted a first token and before any request
finished. The native counter was sampled every 250 ms. C1 comes from the
phase sweep above; C2–C4 have two measured waves each. Tasks differ between
points, so this is a serving-load screen rather than a paired scaling test.

| Input budget / output per request | Load | Fully overlapped aggregate decode | Median batch end to end | MTP accepted | Max waiting |
|---:|---:|---:|---:|---:|---:|
| 4K / 1,024 | C1 | 73.5 tok/s | 16.85 s | 53.1% | 0 |
| 4K / 1,024 | C2 | 110.5 tok/s | 24.21 s | 45.6% | 0 |
| 4K / 1,024 | C3 | 146.9 tok/s | 30.38 s | 45.3% | 0 |
| 4K / 1,024 | C4 | 185.2 tok/s | 34.75 s | 45.3% | 1 |
| 16K / 1,024 | C1 | 65.7 tok/s | 28.08 s | 49.3% | 0 |
| 16K / 1,024 | C4 | 166.7 tok/s | 76.58 s | 52.3% | 2 |

The fully overlapped sampled intervals totaled 35.7, 38.3 and 40.9 s for
4K/C2–C4 and 43.4 s for 16K/C4. No concurrent wave preempted.

## Exact prefix resend

The unique run namespace kept the first 16,383-token prompt cold. The second
request used identical prompt bytes.

| State | Cached / computed prompt tokens | TTFT | End to end |
|---|---:|---:|---:|
| Cold | 0 / 16,383 | 12.54 s | 29.69 s |
| Warm | 13,312 / 3,071 | 2.53 s | 18.65 s |

Raw prompts, response text, streams, metrics and timelines remain in the
ignored local directory `benchmark-results/meaningful-source-profile/run-20260923-193431-w0.00/`.
The tracked summary contains no prompt or response text. To rerun the short
plan on an exclusive engine:

```bash
python3 scripts/current-profile-benchmark.py \
  --scenarios benchmarks/meaningful-short-scenarios.json \
  --output-root benchmark-results/meaningful-source-profile --execute
```

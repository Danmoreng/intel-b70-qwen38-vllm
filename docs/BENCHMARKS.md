# Current benchmark evidence

[Back to README](../README.md) · [Reproduction](EXL3_BENCHMARK_REPRODUCTION.md)

## Serving measurements

The current evidence is the [2026-11-12 cache1792 qualification](../benchmarks/results/mtp-cache-1792-20261112/README.md).
It uses EXL3 4.00 bpw, FP16 activations, FP8 KV, MTP3, 180 W, 16 admission
slots, batch 4096, LBNHC and 12 GiB host RAM cache. The serving matrix contains
20 scenarios, 70 measured waves and 124 requests, with 126,976 output tokens.
All payload/prompt bytes are paired and token/finish accounting passes.
The successful full repeat has zero preemptions. An earlier sweep failed with
the open BCS GPU-memory fault; the passing repeat does not erase that failure.

Sampling is temperature 1, top-p 0.95, top-k 20, fixed per-request seeds,
thinking disabled, `ignore_eos=true`, and 1,024 output tokens. Startup and
warmup are excluded. This is a throughput workload, not an answer-quality test.
The warm-prefix scenarios include a cold request followed by two warm repeats;
only four waves in the complete matrix are warm repetitions.

The [README table](../README.md#measured-performance) is generated from the
candidate medians in [comparison.json](../benchmarks/results/mtp-cache-1792-20261112/qualification/full-serving-20261112-r2/comparison.json).
It includes three cold single-request contexts, two prefix-reuse cases and a
four-request case. The [scenario CSV](../benchmarks/results/mtp-cache-1792-20261112/qualification/full-serving-20261112-r2/scenario-comparison.csv)
and comparison JSON retain all twenty cases, ranges and regressions.

The measured worker used the v6 native image with explicit 1792/128 settings
and the complete COW helper installed. The final v7 child installs that same
helper automatically. [Packaging and exact-child checks](../benchmarks/results/mtp-cache-1792-20261112/README.md#release-packaging-and-deployment)
bridge those identities; the full serving matrix was not rerun on the packaged
child. These are profile-qualification results, not a new measurement of every
subsequent documentation or watchdog change.

## Reading the numbers

- **Time to first token (TTFT):** client latency until the first output. The
  table uses the median across requests within each wave, then across waves.
- **Single-request decode:** post-first output tokens / server decode time.
- **Parallel decode:** combined post-first throughput during the fully
  overlapping interval, after every request starts and before any finishes.
- **Wave time:** end-to-end wall time until the wave's requests complete.
- **Whole-matrix output rate:** total output tokens / summed measured wave
  time, including prompt processing and request overhead. The current value is
  52.53 tok/s, distinct from decode-only rates.
- **Energy:** GPU card counters, including measured prefill/decode and request
  overhead. These do not measure whole-PC consumption.

Against the stored November 10 v6 reference, wave time changes by −0.28%,
weighted native decode by +1.47%, and weighted native prefill by −3.83%.
Warm 16K/64K prefix request time is about 10% shorter; TTFT is 68–77% shorter.
Some cold short cases add 0.1–0.15 s TTFT. The single 199K case has 6.15% lower
decode alongside changed draft acceptance. The reference is historical rather
than same-day randomized ABBA. These results do not establish a universal
speedup or a causal kernel gain. See [review of all flagged cases](../benchmarks/results/mtp-cache-1792-20261112/qualification/full-serving-20261112-r2/trigger-review.json).

## Functionality, capacity and quality

The [tool-agent panel](../benchmarks/results/mtp-cache-1792-20261112/qualification/agent-panel-20261112-r1/comparison.json)
runs two task families with two seeds each, using tools, natural completion,
normal thinking sampling, an 8,192-token thinking budget and independent hidden
tests. Both profiles solve all four cases: eight actual agent runs in total.
Median paired task-time change is −0.65%, effectively a tie on this small panel.
It does not replace a broad coding-quality evaluation.

The [technical gates](../benchmarks/results/mtp-cache-1792-20261112/qualification/TECHNICAL-GATES.json)
cover controlled C1/C2 numerical/state checks, COW copies, the 17,920 boundary,
actual GPU/RAM restore, abort/reuse and admission behavior. The
[full-context receipt](../benchmarks/results/mtp-cache-1792-20261112/qualification/context262144-r1/result.json)
records an executed 261,120-input + 1,024-output sequence. The lifecycle pressure
gate uses a reduced 2 GiB/two-slot pool; it is not proof of sixteen simultaneous
262K histories. Numerical controls are absent from production.

Older Flappy Bird, QueueKit and PPL/KL comparisons retain their original runtime
and failure scope in the [archive](ARCHIVE.md). They are not current-profile
results. A fresh broad coding panel or power-limit comparison can be added as
a separately identified campaign; neither is claimed here. Existing current
profile evidence was reused for the documentation consolidation, with no new
GPU campaign or image rebuild.

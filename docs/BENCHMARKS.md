# Current benchmark evidence

[Back to README](../README.md) · [Reproduction](EXL3_BENCHMARK_REPRODUCTION.md)

## Serving measurements

The current profile is the exact PTI/M04-fixed R12 image, deployed as EXL3 v8
on 2026-11-15. It runs vLLM 0.33, Torch 2.15, EXL3 4.00 bpw, FP16 activations,
FP8 KV, MTP3, 180 W, 16 admission slots, batch 4,096 and 12 GiB host RAM with LBNHC.
[Release approval and deployment](../benchmarks/results/direct-xpu-followup-20261115/release/README.md).

The [full 2026-11-14 campaign](../benchmarks/results/direct-xpu-20261114/README.md)
contains 20 scenarios, 70 measured waves, 124 requests and 126,976 output tokens.
Every payload and prompt is paired with the v7 reference; token/finish
accounting passes and there are zero preemptions. The README table uses the
candidate medians in the [complete comparison](../benchmarks/results/direct-xpu-20261114/qualification/m04-fixed-r12/serving-comparison.json).
This campaign measured the exact final image, including its native libraries.

Sampling is temperature 1, top-p 0.95, top-k 20, fixed per-request seeds, thinking
disabled, `ignore_eos=true` and 1,024 output tokens. Startup and warmup are
excluded. This throughput workload is separate from semantic quality checks.
Warm-prefix scenarios include a cold request followed by two warm repeats;
only four waves in the complete matrix are warm repetitions.

## Reading the numbers

- **First token:** client latency including queueing and prompt processing;
  median across requests in each wave, then across repeated waves.
- **Single-request decode:** post-first output tokens / server decode time.
- **Parallel decode:** combined rate during the interval when every request
  has started and none has finished.
- **Wave time:** wall time until every request in the wave completes.
- **Whole-matrix output rate:** output tokens / summed wave time, including
  prompt processing and request overhead: **51.68 tok/s** for R12.
- **Energy:** GPU card counters, including measured request overhead;
  these are not whole-PC energy measurements.

Compared with the fresh v7 reference, summed wave time is **1.82% longer**,
weighted native decode **0.35% faster**, and native prefill **6.53% slower**.
Two 2K/C2 stalls dominate the total-time loss. Removing those cases would only
localize the issue and must not replace the full-matrix score.

The [2026-11-15 follow-up](../benchmarks/results/direct-xpu-followup-20261115/README.md)
adds 38 measured waves / 50 requests in four counterbalanced sessions, with
fresh workers and their respective existing compilation caches. Both images
are measured fresh and after the same 128K prologue. No 2K/C2 stall recurs in
six new-image repeats (18.31–21.04 s), but rare recurrence remains possible.
4K/C1 retains **+6.07% median time and +2.73% summed time** across ten responses
per image. Lower draft acceptance accompanies nearly unchanged estimated
verification-round cost; this does not prove a specific library cause.

The follow-up supplements the full score. It does not replay every historical
preceding request, prove 12 GiB histories retained, or establish a universal
speedup. The operator explicitly accepted these measured costs for production.
[Review of every flagged trigger](../benchmarks/results/direct-xpu-followup-20261115/release/performance-review.json).

## Functionality, capacity and quality

All nine [final-image stages](../benchmarks/results/direct-xpu-20261114/qualification/m04-fixed-r12/gates/campaign.json)
pass: independent M04 boundaries, profiled native graph replay, original kernel
tolerances, model/state replay, startup, concurrent partial-page COW, natural
tool-agent tasks, operations and full serving. Operations cover media, API,
RAM restore, long-context admission, abort and restart. Configured 16-slot
capacity does not promise 16 simultaneous 262K requests.

The new image passes **4/4 natural tool-agent tasks**, with independent hidden
checks. The fresh v7 reference passes 3/4. Different generated lengths mean task
times are not isolated engine speed measurements. Four tasks do not establish
broad coding quality.

All 16 short and four long teacher-forced windows are finite and aligned.
[Quality comparison](../benchmarks/results/direct-xpu-20261114/qualification/m04-fixed-r12/quality-comparison.json):
short-panel perplexity changes −0.038%, long suffix −0.186%, and full long
prompt −1.253%. Paired-window confidence intervals include zero. Identical
outputs across runtime versions and general accuracy improvements are not
claimed.

The open sporadic BCS GPU-memory fault remains a limit; passing finite runs
and the watchdog do not establish its cause or fix it. Earlier failed images,
original performance outliers and diagnostic hypotheses remain in the dated
receipts. Older Flappy Bird, QueueKit and other release scores are in the
[archive](ARCHIVE.md), with their original runtime scope.

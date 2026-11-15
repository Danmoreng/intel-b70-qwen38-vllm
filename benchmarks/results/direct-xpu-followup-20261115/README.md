# Targeted performance follow-up — 2026-11-15

**The large 2K/C2 stalls did not reproduce; a smaller 4K/C1 cost remains.**
Neither image, sampling settings nor production profile was changed. All four
comparison sessions passed: **38 measured waves, 50 actual requests**, with
25 old/new request pairs verified for identical payloads and prompt bytes,
exactly 1,024 output tokens per request, no KV reuse in these cold cases and
zero preemptions. Warmups and startup smokes are separate from these totals.

The old image is v7 (`dc818253…`), Torch 2.14.0; the new image is the unchanged
M04/PTI-fixed R12 (`131dd9e1…`), Torch 2.15.0. Both run vLLM 0.33, MTP3, native
graphs, 16 slots, 12 GiB RAM and the same model revision at 180 W.

## Paired measurements

These are medians of complete response/wave time. A negative change is faster.
Parallel rows contain two simultaneous requests, each producing 1,024 tokens.

| Condition | Waves per image | Old v7 | New R12 | Time change |
| --- | ---: | ---: | ---: | ---: |
| Fresh worker, 2K/C2 | 3 | 20.54 s | 19.82 s | -3.54% |
| After 128K sequence, 2K/C2 | 3 | 19.34 s | 19.73 s | +1.99% |
| Fresh worker, 4K/C1 | 5 | 17.55 s | 18.34 s | +4.54% |
| After 128K sequence, 4K/C1 | 5 | 17.72 s | 18.83 s | +6.22% |
| 128K/C1 prologue itself | 3 | 126.03 s | 126.66 s | +0.50% |

The six new-image 2K/C2 waves took **18.31–21.04 seconds**; the six reference
waves took 19.25–21.77 seconds. The previous 28.42/61.85-second stalls remain
recorded in the [original full campaign](../direct-xpu-20261114/README.md).
This follow-up does not establish that a rare stall cannot recur.

## What explains the remaining 4K difference?

Across both conditions, the 4K median is **17.63 → 18.70 seconds (+6.07%)**.
Summing all ten responses gives **180.65 → 185.59 seconds (+2.73%)**. These
statistics answer different questions; the median change is not a 6% increase
in the total time of every workload.

The recorded MTP counters help distinguish prediction behavior from step cost:

| 4K/C1 diagnostic, both conditions | Old v7 | New R12 |
| --- | ---: | ---: |
| Weighted accepted draft-token fraction | 55.26% | 52.37% |
| Weighted native decode | 63.33 tok/s | 61.37 tok/s |
| Median estimated verification-round time | 41.91 ms | 41.95 ms |
| Median time to first token | 1.88 s | 1.86 s |
| Weighted native prefill | 2,147.71 tok/s | 2,176.53 tok/s |

For each image, all five 4K output hashes repeat exactly between the fresh and
post-128K conditions. Across images, the output texts differ, despite identical
inputs and configured seeds. The lower draft acceptance in the new outputs
is consistent with the lower token throughput, while estimated round cost is
nearly unchanged. This is evidence about the observed behavior, not proof of
one specific library or kernel as its cause. Round estimates include host and
device costs and are not isolated GPU-kernel latency measurements.

## Method and limits

The [contract](frozen-contract.json) was frozen before GPU measurements. Order:
R12 fresh, v7 fresh, v7 after 128K, R12 after 128K. Each controller uses a fresh
worker and its own image's existing compilation caches. Scenarios retain the
original warmups, prompts, seeds, output caps and acceptance assertions. The
[scenario-order option](../../experiments/vllm-033-upgrade/run-candidate.py)
only selects/reorders cases; the original full-matrix default is unchanged.

The pressure prologue replays the original 128K warmup and three measured
requests. Both arms transfer about 19.91 GB GPU-to-CPU before the target case;
[the counters](offload-prologue-evidence.json) establish the exercised traffic.
They do not mean that 12 GiB remained retained. The entire historical 36-wave
prefix was not replayed, and arbitrary previous workloads are not covered.

Each condition has one worker session per image and a small fixed fixture set;
the ten 4K observations include the same five inputs in two conditions. This
is a targeted follow-up, not a new full-70-wave score or a general performance
guarantee. No slow sample was removed. The separate numerical, quality,
agent and operational results on the unchanged R12 image remain recorded.

## Result and operational state

The 2K/C2 result no longer indicates a persistent general slowdown in this
follow-up. **The 4K/C1 latency cost remains a documented tradeoff.** No dependency
was rolled back or new image built on the basis of these measurements.

All test workers have stopped. The original v7 production service was restored
once after all timings and checked with a live generated response. Production
configuration is unchanged; the new profile has not been promoted or published.
See the [final operational receipt](operational-final.json).

[Full numerical comparison](comparison.json),
[within-image output checks](within-image-output-repeatability.json),
[measurement receipts](cases/) and
[reproducible validator](../../experiments/vllm-033-upgrade/summarize-performance-followup.py).
Raw requests, streams, metrics and full worker logs remain in the local
`benchmark-results/direct-xpu-followup-20261115` tree.

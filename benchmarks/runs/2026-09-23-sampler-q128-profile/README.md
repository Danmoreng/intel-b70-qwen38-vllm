# Short sampler operator screen and Q128 profile

Measured 2026-09-23 on the B70 at 180 W with the promoted vLLM 0.30.0+xpu,
Torch 2.13.0+xpu and XPU-kernels 0.1.15.4 image
`sha256:648132c9b9da4bb244d7304b956c1a9bb825be640a92755a5ffe32f2bbd679b4`.
The exclusive GPU lock was held, the idle production service was stopped during
the isolated probes, then restored and checked for health, image ID and power
cap. No candidate image or production code was changed.

## MRV2 sampler operator

The current V2 runner filters logits with `apply_top_k_top_p` and samples with
Gumbel. The alternative called the published fused XPU operator directly with
the [upstream MRV2 integration's](https://github.com/vllm-project/vllm/pull/57277)
generator-state handling. Both arms used FP32 logits of shape `B × 248320`,
`top_k=20`, `top_p=0.95`, the same input logits and an identical input copy in
the timed region. There were four warmups and eight alternating ABBA/BAAB pairs
per batch size. Medians are XPU event times; wall medians are in
[`summary.json`](summary.json).

| Batch | Current V2 path | Fused XPU op | Device time saved |
|---:|---:|---:|---:|
| 1 | 0.959 ms | 0.418 ms | 0.541 ms (56.4%) |
| 2 | 1.285 ms | 0.575 ms | 0.710 ms (55.2%) |
| 4 | 0.684 ms | 0.311 ms | 0.373 ms (54.5%) |

All observed samples were within the top 20 logits. The fused arm produced
multiple distinct token IDs on each row. This is an **operator timing and basic
support screen**, not a distribution-equivalence or full-model quality test.
The two RNG implementations need not return identical IDs. Upstream's fused
MRV2 path deliberately falls back to native sampling for explicit per-request
seeds, so this gain is relevant only to eligible unseeded requests. The earlier
seeded Wikipedia A/B would take the fallback. The published vLLM 0.30.0 image
does not yet route its V2 runner through this fused operator; this result is not
a measured serving speedup.

**Decision:** The approximately 0.4–0.7 ms operator saving merits a short
C1/C4 serving A/B once a published vLLM tag contains the MRV2 integration.
Measure unseeded production-like sampling, while keeping the separately seeded
correctness/variability arm. Do not infer a percentage end-to-end gain from the
operator percentage.

## Q128 at long context

Q128 and native XPU attention received identical FP16 queries, paged FP8 K/V,
causal mask, descales and the production interleaved K/V stride
`(3407872, 2048, 512, 1)`. Query length was 6,656. Each arm had three warmups
and four alternating pairs. Outputs matched **exactly** at both lengths.

| KV length | Q128 | Native | Q128 advantage | Device-wide bandwidth | Power |
|---:|---:|---:|---:|---:|---:|
| 64K | 285.82 ms | 313.31 ms | 9.6% | 1.0% | 173.7 W |
| 192K | 911.58 ms | 959.43 ms | 5.2% | 2.7% | 176.7 W |

PyTorch's XPU trace recorded exactly one `XeFMHAFwdKernel` dispatch per Q128
call at both lengths. Thus there is no separately dispatched softmax kernel to
optimize in the way described for llama.cpp. `xpu-smi` supplied card-wide
bandwidth, power and frequency while only Q128 ran. It did not expose EU,
occupancy, GRF, spill or per-stage memory counters, so these readings do not
establish whether Q128's **internal** softmax/reduction is the bottleneck.
The trace's three device durations per length and the raw telemetry remain in
the ignored local run; compact values are in [`summary.json`](summary.json).

**Decision:** Keep the deployed Q128 kernel. Its performance agrees with the
earlier [long-context sweep](../2026-09-23-q128-long-context/README.md), and
this profile gives no evidence for a specific kernel rewrite. A further Q128
change needs source-level GPU counters or a controlled stage-isolation probe,
then an isolated operator gate before serving.

Runner and probe scripts: [`m20-sampler-q128`](../../experiments/m20-sampler-q128/).

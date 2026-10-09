# Pro work-order decision: retain MTP3 for general serving

Completed on 2026-10-10 (Europe/Berlin). The first infrastructure-only commit
was `d857bd1`. The bounded experiments do not justify a serving promotion.
The qualified production image `sha256:194a8786f12065e65647d914ddc7fdc136bafbeec73f476730ef1324e4d2c7ec`
is restored and healthy, with FP16, FP8 KV, MTP3, 262,144 total tokens,
`max_num_seqs=16`, `max_num_batched_tokens=4096`, memory utilization .965 and 180W.
This is the best choice within the tested candidates, not a global optimum.

## Direct performance evidence

The frozen panel contains 512 output tokens per request, T=1.0/top-p=0.95/top-k=20,
the same input IDs/seeds, and separately verified cold and warm prefix states.
The predeclared workload weights are 35% 32K C1, 35% 64K C1 and 30% 32K prose C4.
MTP3/4 use the same wide-M04 experiment image. DFlash uses group-four W4 images;
these controls are separate from the published full production benchmark.

| Frozen cell, tok/s cold / warm | MTP3 reference mean | W4 DFlash full QKV | W4 DFlash K/V-only |
|---|---:|---:|---:|
| 32K C1 | 56.41 / 55.93 | 53.05 / 52.95 | 52.52 / 52.44 |
| 64K C1 | 47.44 / 47.45 | 44.53 / 43.23 | 49.30 / 41.57 |
| 32K prose C4, client aggregate | 32.17 / 101.01 | 25.73 / 59.12 | 25.67 / 58.60 |

C1 cells use native request-weighted decode; C4 uses aggregate client decode
intervals. Rates exclude the first output per request. MTP3 entries above are
the mean of two reference arms; each DFlash projection has one later arm.
Different time/order and stochastic output trajectories make the DFlash panel
observational. It has no paired confidence interval and cannot qualify promotion.
All waves complete with zero observed preemptions and zero excess recomputes.

The bounded MTP3/4 ABBA result is −0.53% weighted improvement for MTP4
(+0.11% cold, −1.16% warm). Its descriptive 95% paired interval is
[−3.57%, +2.52%], from only two independent AB/BA blocks. Long C1 cells gain
about 4–12% at 32K and 1–8% at 64K. Warm C4 aggregate decode instead regresses
17–19%, and mean end-to-end latency rises 20–22%. MTP4 remains an interesting
unpromoted specialist for long single requests; this mix does not pass the
5% gain or C4 guards. No further MTP depth matrix was run.

W4 full-QKV DFlash is −5.56% on the weighted native panel, and K/V-only is
−4.97%. K/V-only has a promising cold 64K cell (49.30 tok/s versus 47.44), but
its warm cell drops to 41.57. Both fail C4 throughput/latency guards: warm
client aggregate decode is 41–43% below the two MTP3 controls. Native C4
rates alone obscure queue and prefill overlap; they must not be multiplied
by four to claim aggregate performance. [Complete panel analysis](panel-comparison.json)
and [paired MTP analysis](mtp4-bounded/analysis.json) retain the separate metrics.
The earlier three-repeat 1,024-token screens retain every outlier and do not
supply a stable gain absent in the matched panel.

## Context and memory

The W4/G128 draft is real packed oneDNN W4A16; the target remains EXL3.
Its logical parameters are approximately 1.193 GiB. Deduplicated registered
parameter/buffer storage is 1.255 GiB for full QKV and 1.256 GiB for K/V-only,
excluding the aliased target embeddings/full head and separate speculator,
KV, allocator and graph buffers. No dense context-KV weight copy is retained.
Source BF16 small tensors are converted offline once; target/draft serving
is FP16 with permitted internal FP32 accumulation and no BF16 bridge.

Group size refers to KV-cache layers per shared pool group, not proposal
count: the draft remains B8/V7. Group five creates 15 groups (10 Mamba / 4 full / 1 SWA);
group four creates 18 (12 / 4 / 2). At 192K the reconstructed padded pool requirement
falls 8.918 → 7.465 GiB, with padding 1.530 → 0.076 GiB and the live-layer/reservation
requirement unchanged. Both group-four images passed an actual 190,976-token
input plus 1,024 outputs and a separate 191,984 + 16 near-limit stochastic/one-token
custom-stop check. Thus 192,000 is actually tested, not merely a startup flag.
Larger allocator-derived limits and the old 166,400 estimate are unqualified.
The user's 166K speed-dependent fallback was not needed.

At the same 96K limit, all four 256-token greedy cold/warm outputs are identical
across grouping variants. Group-four rates are 0.34 / 0.80 / 0.37 / 2.48% lower;
the largest difference includes 86 versus 84 proposal rounds. This does not
establish zero performance overhead. [Matched grouping evidence](grouping-matched96.json)
retains the stochastic repeats and a short CPU-reference overlap caveat.

Configured `max_num_seqs=16` is preserved, but pool capacity still limits long
request admission. Both final DFlash panels periodically show Running=3 / Waiting=1;
no periodic sample shows Running=4. Their pools have 731 / 729 blocks at 13 MiB,
versus 627 in the initial instrumented capacity capture. The actual-spec
reservation model needs 124 bounded-group blocks plus 80–84 full-attention
blocks per 32K request: four need816–832 blocks. This is consistent with
admission pressure, not proof of an exact allocator ceiling. Prefix residency
and scheduling also matter. Four submitted C4 requests are not automatically
four simultaneously decoding requests; max context and concurrency compete
for the same pool.

## Correctness and cost evidence

Actual eager capture passes 244 frozen-bound comparisons over four selected
rounds: post-residual feature taps/order, packed FC/KV, norms, convolution,
selector, RoPE with actual stored coefficients and paged attention with actual
stored FP8 KV. Twenty-two small-input norm probes exercise epsilon 1e-6 / 256.
All 21 W4 runtime modules / 36 checkpoint members pass independent selected-column
FP64 packed dots and strict role/inventory checks. All five K/V-only projections
match full packed QKV exactly at 8, 64 and 4,096 rows. Shared head/storage aliases
and no dense fallback are attested.

Actual greedy acceptance/residual/bonus decisions pass independent scalar
Philox/Murmur/Gumbel replay. Three published Random123 known-answer vectors
check the CPU RNG reference. Older probabilistic captures independently check
q/acceptance but lack an attested Gumbel-mode winner check. Statistical exact
vocabulary tests, changed-slot proposal/sampler graph replay and synthetic
C1/C4 paged-attention page/SWA/stale-tail gates are separate from actual eager
branches. **Ideal FP64 full RoPE rotations fail the original bounds at 32K**;
cache-theta and native rotations with actual FP16 coefficients pass their
separate frozen gates. Failures and ideal errors are preserved, not relabeled.
No actual graph-internal branch equivalence or broad semantic qualification
is claimed. [Actual eager evidence and failed attempts](actual-eager/branch-replay.json)
record these limits.

The unchanged B8/V7 diagnostic spans are 62.28 ms whole round, 48.08 ms target
body, 2.05 ms target head, 0.68 ms verification sampling and 11.02 ms complete draft.
They include dispatch gaps and are non-additive medians. Eliminating the entire
draft gives an approximate 17.7% time-saving ceiling (ideal speedup 1.215).
The separate kernel trace is dominated by target EXL3 GEMMs and M04. Ambiguous
draft GEMM attribution is not guessed; the selector is too small to justify
a large rewrite. [Phase-A cost evidence](../dflash2-pro-diagnostics-v031/DECISION_A.md)
retains the original graph profile and sources.

Static B8/V4 is **not implemented or measured**. The conditional branch was
not triggered by the long-C1 profile: truncating observed prefix counters
needs 14–17% round-time savings, while target C1 DPAS remains MB8 and M04
query tile8 for both five/eight rows, and the draft backbone still computes
all eight. This is a stop decision, not proof V4 is universally slower.
C4 MB24 versus MB32, 64K attention split differences, reduced head work and
smaller target speculative-state reservations could help but remain unmeasured.
No adaptive flag, blind draft marker change or GCD=1 grouping shortcut was used.

Original-target teacher-forcing and causal task/thinking effects remain open;
dequantized EXL3 weights would not isolate original quantization loss. No relevant
same-stored-KV attention error triggered a separate FP16 draft-KV experiment.

## Delivery and retained policy

Keep MTP3 as the general serving default. Preserve MTP4 long-C1 results and the
working 192K quantized DFlash implementation for a future focused hypothesis;
current evidence does not support further depth/adaptive experimentation or a
production DFlash switch. Existing qualified production benchmark and historical
power tables remain intact: 180 W was selected for performance per watt, and
power comparisons were not remeasured.

[Machine-readable decision](decision.json), per-campaign source/image hashes,
rollback receipts, raw manifests and [publication manifest](PUBLICATION_MANIFEST.json)
make the evidence reviewable. Large tensor/SSE captures and immutable executed
source snapshots stay locally under `benchmark-results/`; public manifests
retain paths and SHA256 values. Current helper/recipe bytes and historical
executed image bytes are distinguished; rebuilding newer recipes requires
new qualification.

Public JSON represents non-finite diagnostic log margins/winner gaps as
explicit strings. [Serialization receipts](json-normalization.json) retain
the original and published hashes and each changed field; immutable raw
records and measured decisions are unchanged.

# EXL3 production release qualification — 2026-10-02

## Decision and scope

**Release approved and deployed: the qualified EXL3 profile is active and enabled.**
The user selected the EXL3 quality/context tradeoff, authorized the qualified
release and final README benchmark/push, accepted genuine C16 pool pressure,
and limited kernel work to a brief check. Short-C1 decode parity is not claimed.

All mandatory engine gates A–H pass within the documented scope. The generated
QueueKit program has a real failed acceptance case: **1/2 tasks, 7/8 tests**.
The native-attention control has the same failed case. These are preserved
utility outcomes, not passed tests or repaired submissions. The quality gate
covers precision fidelity and the production API/tool/parser/media contracts;
phase 19 separately requires recording coding success and failure categories.
Both corrected Flappy v7 outputs pass **54/54**. One adaptive pair cannot
establish a general coding-quality ranking.

## Exact artifacts

- Runtime image: `sha256:6cf4899923749b221a008834f7fd42e0aa7e3f58546f408faf77cbf4da86554b`.
- Measured release image / production alias: `sha256:c09015ce22180fbc90ef0f5070f4a7116c8d11be785067499477accf0216f21f`, `local/b70-qwen38-vllm:production-exl3-v1`.
- Policy: `cbe1755c37b888528797ca5c0f51cf9562aa4bd8cb8ee149d0cd7bb15eb78542`.
- EXL3 checkpoint: `turboderp/Qwen3.8-27B-exl3`, revision `113cf7ab958054860e43fb7f3063b1af19171095`, 4-bpw body / 6-bpw full target head.
- Native library: `48f879c16f3695dbb0fae7388f1e318591d6408acc7be15b0093a0bbb6d9053b`.
- M04 library: `eaa18427db27d4fceeca8a17c8a3c6b019b7678f39bf98affc1736b9f2c2d631`.
- oneDNN library: `ee8fc42adcf851ea0f8013ebda24f07629f49ca6ed815835a16da45ad718ca20`.
- EXL3 source snapshot: upstream `15ded2f3add148c4db3c900cba7de878238f53bc` to migration `81ced069d3953b8f7d320aadfdb50808ec44c771`. Later harness-only changes are distinguished from the frozen runtime build.

The release metadata child has identical RootFS to the tested runtime. Alias
promotion does not rebuild it. [Canonical manifest](../config/production_image.json)
records all 11 checkpoint-file hashes, middleware, profile and native artifacts.
[Complete tracked EXL3 source and patch](../engine/exl3xpu/README.md), M04 source
and pinned build/header manifests are published without weights or large arrays.

## Applied profile and implementation

vLLM 0.30.0 / Torch 2.13.0+xpu / oneAPI 2026.1.1 / SYCL 9 / oneDNN 3.13.0;
Intel Runtime 26.35.39758.10 and IGC 2.41.5. One B70, 180 W, memory fraction
0.965, 262,144 total tokens, 16 admissions, clean moderate C4 qualification,
4,096 batched tokens, full-ISL and watermark 0, FP8 KV and Mamba-aligned prefix
reuse. FULL_DECODE_ONLY captures 1/2/4/8/12/16/24/32/40/48/56/64 rows.
MTP3 retains 65,536 draft rows and the full 248,320-row target head.
Tool/reasoning parsers remain qwen3_xml/qwen3. Limits are 32 images / 4 videos
within total context and 4,194,304 pixels per image; default output cap 16,384.

Builds fail closed and attest source/ABI/library identity. Loader discovery uses
the authoritative weight index and validates fused/head/MTP transforms. GPTQ
MTP overrides are format-scoped; discarded duplicate full draft-head ownership
is released. Native EXL3 row dispatch is retained. The rebuilt M04 shares KV
across supported verification queries. Guarded exact-K oneDNN prefill uses
once-per-step V2 metadata with native GPU decode lengths and owns no extra KV
updates. Constant scales/partitions are owned by the actual thread/stream.

## Correctness and quality gates

| Gate | Result | Evidence |
|---|---|---|
| A artifact/build | PASS | Fail-closed fresh builds, ABI/source/library checks, immutable RootFS attestation and strict launch preflight |
| B loader/tensors | PASS | 409/409 reconstructed tensors, including 8 MTP; old/target ABI and actual serving inventory |
| C kernels/shapes | PASS scoped graphs | Seven classes/full head, 18 row counts 1–512, RMS 2e-3 unchanged, tail poisoning and large-first compilation; SmallM graphs through 128 |
| D attention | PASS | Exact/page/mixed/causal numerical tests, 31 packaged XPU cases, two mutable graph replays, three future-KV poison controls; unsupported native fallback |
| E precision/API quality | PASS finite panels/contracts | Matched native/optimized arrays, BF16 suffix checks, generated-route diagnostics and API/tool/parser/media integration; failed generated QueueKit case remains separately scored |
| F capacity/stability | PASS accepted C16 pressure | Exact 262K boundary, clean C4, 32 unique max-area images, videos, prefix extension, confirmed abort/recovery and restart |
| G performance | PASS documented tradeoff | Controlled MTP decision, long ABBA, mixed tails and full frozen C1–C4 matrix; residual short decode deficit remains visible |
| H rollback | PASS | Independent pinned GPTQ launcher/image actually served and completed the corrected Flappy task; incompatible caches separated |

Original BF16 short PPL is 3.604529; historical GPTQ PPL/KL is
3.801927 / 0.086369. Final EXL3 is **3.646722 / 0.032481** (about 62% lower
KL on this finite panel). Matched 128-row native/optimized capture produces
272 bit-identical short arrays. Engineered 32K/100K/180K/262K-prefix suffix
windows produce PPL **1.208579** versus BF16 1.210748, KL **0.001178**, and
32/32 sampled top1 agreement. These are suffix metrics, not whole-prompt PPL.
Existing BF16 arrays were integrity-checked and reused.

Twenty generated requests / 2,560 tokens per arm show nine differing histories.
Every optimized choice is a reported logits maximum; recorded margins are not
uniformly tiny. No bit-exact-text claim follows. Finite panels, teacher-forced
capture batching and tested shapes bound the evidence.
[Quality receipts and limitations](../benchmarks/results/exl3-migration/optimized-quality-v1/README.md).

## Capacity and preemption

261,120 input + 1,024 output reaches exactly 262,144 total without preemption.
139K/188K and C4×32K are clean. All 16 independent 8K+256 requests complete;
four preemptions affect two requests, with 16,000 extra **submitted** prefill
rows. Aligned Mamba/GDN and speculative states share the block pool; logical
attention tokens alone do not describe occupancy. C16 pressure is accepted,
not mislabeled clean throughput or 16 simultaneous maximum contexts.

32 distinct maximum-area images produce 131,164 input tokens without cache
hash deduplication or preemption. Long-image, video-limit/over-limit, prefix,
scheduler-confirmed FINISHED_ABORTED recovery and fresh restart pass.
[Operational receipts](../benchmarks/results/exl3-migration/optimized-operations-v1/README.md).

## Performance and full README benchmark

The immutable final image completed **20 scenarios / 70 waves / 124 requests**
in **53.69 minutes**, versus the published GPTQ run's 53.72 minutes. All 248
original fixture-file hashes/payload values validate; each request emits exactly
1,024 nonempty output tokens, with zero preemptions/recompute. Admission is 16;
the frozen measured matrix stays C1–C4. GPTQ's full matrix was not rerun.
Three additional isolated 64K resends supply the cold prefix row: 39.09 s cold
TTFT to 2.48–2.49 s warm, 62,400 of 65,469 input tokens cached.
[Validated final summary](../benchmarks/runs/2026-10-02-exl3-production/summary.json).

| Context budget | GPTQ decode tok/s | EXL3 decode tok/s | Change |
|---:|---:|---:|---:|
| 479–507 actual | 68.12 | 58.21 | -14.6% |
| 1,992–2,047 actual | 73.06 | 59.72 | -18.3% |
| 4,052–4,094 actual | 67.44 | 55.19 | -18.2% |
| 8,167–8,186 actual | 62.98 | 54.76 | -13.1% |
| 16,335–16,379 actual | 68.45 | 56.12 | -18.0% |
| 32,704–32,762 actual | 62.23 | 56.53 | -9.2% |
| 65,491–65,532 actual | 52.52 | 52.69 | +0.3% |
| 131,034–131,070 actual | 44.78 | 39.90 | -10.9% |
| 199,673–199,673 actual | 37.17 | 38.49 | +3.5% |

C4 aggregate fully overlapped decode is 203.73→185.62 tok/s at 2K (-8.9%),
196.63→181.56 at 4K (-7.7%), and 162.79→168.21 at 16K (+3.3%). Same payloads
and seeds do not produce identical histories across different checkpoints.
The remaining short-C1 deficit is the documented quality/context tradeoff,
not hidden behind the adaptive task's wall-clock improvement.

Repeated whole-attention ABBA at 103K C1: 43.98→49.24 cold and 43.91→49.24
warm (~12%), equal cache residency, stable repeats within each arm. Histories
differ from token 8; this is whole-pipeline serving evidence. The separate
M04-only proof shows ~14% with identical 512-token outputs. Mixed ABBA's
incoming 49K TTFT improves ~36→27 s; overlapping SSE gap p95 ~3.0→2.05 s,
zero preemptions/cache hits. MTP4's ~1% long gain did not justify its short-C4
regression; 36-wave/84-request screen retains MTP3. Full vocabulary ablation
(60 requests) slows all 12 cells 1–19%; keep 65,536-row draft pruning.
[Optimization receipts](../benchmarks/results/exl3-migration/optimized-performance-v1/README.md).

## Corrected coding utility

| Flappy v7 | GPTQ production v2 | Final EXL3 |
|---|---:|---:|
| Complete session | 32 min 6 s | 24 min 12 s |
| Model requests | 77 | 66 |
| Maximum actual context | 122,078 | 90,444 |
| Generated tokens including reasoning | 95,473 | 67,555 |
| Native prefill / post-first decode tok/s | 1,207.9 / 58.11 | 1,388.7 / 53.93 |
| Independent functional checks | 54/54 | 54/54 |
| Preemptions | 0 | 0 |

Same six-stage task, sampler/seeds, retained reasoning, 4K thinking budget,
32-request/stage and 40-minute limits. EXL3 uses less generated history and
finishes sooner despite lower overall decode rate. It does not measure agent
decode above 100K; GPTQ does. Empty 10K bands remain unmeasured. Fixed source
sweeps independently cover long EXL3 contexts. No padding or repaired output
is introduced. [All bands, scored outputs and plot](../benchmarks/runs/2026-10-02-flappybird-v7/README.md).

QueueKit v2 optimized: 7 min 36 s, 29 requests, 716–33,967 input context,
1/2 tasks and 7/8 cases. The snapshot restore wrongly requires a saved running
job to occur in the pending-ID list. This violates the supplied contract.
Native-attention control (only EXL3_GUARDED_ATTN=0) is 2 min 25 s, 11 requests,
also 1/2 and 7/8, failing the same roundtrip/detachment case for the same reason.
The historical GPTQ task scored 8/8. Both failed EXL3 programs and test outputs
are retained. This single pair supplies no causal evidence that guarded
attention introduced that semantic failure; it also does not establish general
coding superiority. The explicit failed-task export marks the failure and never
turns it into a passed test or automatic release approval.
[Native control and full failure output](../benchmarks/results/exl3-migration/release-v1/queuekit-control.json).

## Limits, kernel scope and phase completion

All plan phases 0–11 and the measured release work are complete. Phase 12 is
complete within the user's narrowed scope: three 128-token event waves and
two eight-cycle pure-decode traces, no speculative Trellis rewrite. Captured
body-graph internals remain opaque; timing is not a kernel attribution or proof
of an optimum. The target body dominates the bounded cycle spans; head/sampler
and commit are smaller. No further kernel tuning is part of this release.

Native fallback applies to unsupported masks/scales/sinks/windows/parallelism,
adaptive verification and unsupported shapes. Large INT8-prefill graph capture
is unsupported and outside FULL_DECODE_ONLY max 64; large eager/compiled rows
pass. Only this one-card profile is qualified. Finite panels and one task pair
cannot prove correctness/quality for every unseen workload.

## Production and rollback

The strict launcher checks ordinary Python execution, policy/image identity,
all pinned checkpoint files, native/M04/oneDNN/profile/middleware hashes,
180 W power and exact arguments/environment. Compiler caches are isolated by
policy plus image, including neo. Final measured candidate and production
alias have identical immutable payloads; no full benchmark rerun is needed for
an alias. The [actual production receipt](../benchmarks/results/exl3-migration/release-v1/promotion.json)
confirms the same measured arguments/environment, 409/409 modules including
8/8 MTP, full 6-bpw target head and real text/tool/image responses (Paris,
add(13,29), Red). The service is active and enabled.

GPTQ remains `sha256:ed1ebca756abb0e0832d11cd0db026dd7e86df094c6903efe7ae8afbdc290b68`,
policy `4ce5f3bd77710fe08ac4b96ee761c50eb13c2d3b48ac74f20ab7fced82b4b368`.
Its independent launcher actually completes Flappy v7 (54/54), using its own
pinned cache/profile. Stop the EXL3 service and use
`scripts/run-server-gptq-rollback.sh`. For persistent rollback, point the user
unit's ExecStart at that launcher, daemon-reload and restart; verify health,
image/policy and real chat/tool calls. [Exact rollback files/procedure](../config/releases/gptq-onednn-v2/README.md).

[Machine-readable engineering decision](../benchmarks/results/exl3-migration/release-v1/decision.json),
[port history](EXL3_PORT_NOTES.md) and [original plan](EXL3_PRODUCTION_MIGRATION_AND_OPTIMIZATION_PLAN.md)
preserve failed attempts and commit boundaries.

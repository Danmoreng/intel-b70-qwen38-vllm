# EXL3 v2 release qualification — 2026-10-02

The bounded Pro-review changes are qualified and deployed. This release adds
the completion-aware exact-shape partition cache, C4-only direct M04 output
copy and one guarded route decision. Quantization, MTP3, draft vocabulary,
attention formulas, split table and toolchain remain frozen.

## Exact release identity

- Production image: `sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918`, alias `local/b70-qwen38-vllm:production-exl3-v2`.
- Reviewed parent: `sha256:0e711fea1f9231a25289d812fffbde51ed93cbe7bad16c34f7fde3edf3d91737`; filesystem layers identical.
- Explicit partition-cache capacity: 64 per thread/queue. The metadata child
  adds the policy identity and explicit cache environment, without a native rebuild.
- Policy: `1bf624713cff3583148a71c3b4fb9189268cc5d4a47afca03662ce25be3e7839`.
- EXL3 source: `c59d9442aba8610188837e37724600f1517d7335`; native library
  `e60e499aefa95b290924ed98545eb815aca9d0fb1f3ede39b37d1ad49f3b522d`.
- M04 library remains
  `eaa18427db27d4fceeca8a17c8a3c6b019b7678f39bf98affc1736b9f2c2d631`.
- vLLM 0.30.0 / Torch 2.13.0+xpu / oneAPI 2026.1.1 / oneDNN 3.13.0, 180 W.

The [release assessment](../benchmarks/results/exl3-review-release-v2/assessment.json),
[decision](../benchmarks/results/exl3-review-release-v2/decision.json) and
[actual service startup/restart receipt](../benchmarks/results/exl3-review-release-v2/promotion.json)
preserve identities, configuration, source hashes, tests and limitations.

The first cache-diagnostic worker failed before model startup because a file
mount targeted a read-only parent directory. Its failed campaign receipt is
retained. Only that diagnostic phase was rerun with a separate read-only
diagnostic directory; already passed serving, operating and quality gates
were preserved under their original identities. No image bytes changed.

## Fresh serving measurements

The existing 20-scenario / 70-wave / 124-request matrix ran once on the final
deployable image, with identical frozen prompt bytes and request settings.
It completed in 53.55 minutes with zero
preemptions or excess recomputed prefill tokens. A separate fresh-worker 64K
prefix resend preserves the cold/warm observation. Adaptive Flappy, GPTQ,
MTP/vocabulary studies and original BF16 inference were not rerun.
See the [new serving summary](../benchmarks/results/exl3-review-release-v2/serving-summary.json).

The matched v1/candidate screen uses both intended optimized profiles, common
frozen performance windows, fixed temperature/seed and separate warmup.
Short C1, 103K C1, short C4, 32K C4 and mixed C4 are measured. Review triggers
are >5% slower decode/TTFT or >10% higher mixed client p95 burst gap; only an
initially triggered case is repeated once, in reversed image order. These are
bounded decision triggers, not an SLA or a statistical distribution.

| Case / pass | Decode change | TTFT change | Mixed p95 gap change | Trigger |
|---|---:|---:|---:|---|
| short-c1 / initial | +0.79% | +0.05% | — | False |
| long-c1 / initial | -0.01% | +0.25% | — | False |
| short-c4 / initial | +0.49% | +0.06% | — | False |
| moderate-c4 / initial | -0.14% | +0.36% | — | False |
| mixed-c4 / initial | — | +0.15% | -0.01% | False |

No unresolved repeated trigger remains. Candidate microcopy measurements
around +0.8% at short C4 are not published as serving throughput gains.
Equivalent serving performance plus bounded resource retention can justify
this release; no universal speedup is claimed.

## Operating and numerical gates

Host unit discovery: **17 passed, 3 skipped** because they require a pinned
B70 runtime image. [The test log](../benchmarks/results/exl3-review-release-v2/unit-tests.log)
records each case. The image-specific numerical and operational gates below
were executed separately on the actual B70.

- Fresh load: 409/409 modules, including 8/8 MTP; all 13 active source guards
  and their complete runtime-source capture pass.
- API: generated/streaming text, automatic tools, reasoning, image/video
  count limits and capped images pass. Permanent-service checks include real
  default-reasoning requests, exact image/policy identity and a service restart.
- Exact context boundary: 261,120 input + 1,024 output, no preemption.
- Moderate C4, long-image context, prefix extension, scheduler-confirmed
  abort/recovery, independent maximum-area images and worker restart pass.
- C16 pressure: all 16 independent 8K + 256-output requests complete;
  4 preemptions are permitted and recorded. This does not promise
  sixteen simultaneous maximum contexts or preemption-free C16.
- Fresh short-reference smoke: 16,368 scored
  positions, PPL 3.646722, mean BF16-relative KL
  0.032481; all 32 aggregate arrays bitidentical to
  v1. Original reference data is reused. This short panel alone does not
  exercise long oneDNN prefill or prove MTP graph correctness.
- Fresh existing 32K-prefix suffix smoke: 128 scored
  positions / 8 full distributions, PPL
  1.194194 versus BF16 1.184646,
  mean KL 0.000957. It is a compact engineered window, not a new
  broad model-quality campaign.
- Separate fresh candidate matched-state probes exercise actual long prefill,
  mixed serving and true C4 verification graphs. Every local comparison passes
  the original rtol 0.01 / atol 0.003. Histories/states are restored within
  each finite probe; no historical bit-exact-text or universal determinism claim.

## Actual serving-worker cache observation

A separate instrumented worker performs mixed decode/incoming 49K prefill and
70 varied exact prompt lengths plus repeats. Counters are read on the actual
inference thread/current queue after its native attention enqueue. This is
not a separate-process cache query. Each observed cache stays at/below 64,
with hits, misses and eviction; pressure waits, compilation microseconds,
RSS and XPU allocated/reserved memory are preserved in the assessment.
These instrumented times are excluded from the throughput comparison.

| Worker context | Peak entries | Hits | Misses | Evictions | Pressure waits | Cumulative compile | RSS range |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 7937 | 87 | 23 | 0 | 56.288 ms | 4.311–4.326 GiB |

Zero recorded pressure waits does not exercise the GPU all-entries-busy branch;
the separate host policy test covers that branch. The
[147 raw worker observations](../benchmarks/results/exl3-review-release-v2/cache-observations.jsonl) preserve
allocated/reserved XPU memory and the individual cache samples.

The cap applies to each application thread/queue context. It does not globally
bound arbitrary thread/queue counts, oneDNN internal caches or every allocator,
and finite measurements do not guarantee permanent freedom from OOM.
Client p95 SSE burst gaps are delivery observations, not GPU token-step times.

## Historical coding results and rollback

The v1 Flappy/QueueKit runs remain [separately dated historical evidence](EXL3_CODING_BENCHMARKS.md).
QueueKit's 7/8 checks and 1/2 tasks, including the same native-control failure,
remain visible. No coding task was rerun or relabeled for v2.

The immediately usable [EXL3 v1 rollback](../config/releases/exl3-v1/README.md)
retains image `sha256:c09015ce22180fbc90ef0f5070f4a7116c8d11be785067499477accf0216f21f`, its immutable tag, policy, original release
receipts, source and runtime snapshots and compiler namespace. The matched
v1 worker loads and serves real requests during this release qualification.
The strict rollback preflight is separately checked. A failed deployment
automatically restores v1 configuration and the local environment.
No sudo, driver or system-service installation was needed.

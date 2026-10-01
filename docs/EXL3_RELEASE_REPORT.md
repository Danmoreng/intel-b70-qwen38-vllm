# EXL3 release qualification — in progress

Production decision: **do not migrate yet**. GPTQ remains pinned. This is a
progress ledger, not a completed qualification or a recommendation to release.

| Gate | Status | Evidence / remaining work |
|---|---|---|
| A — build/artifact integrity | PASS for final frozen runtime | Pinned Torch-2.12 and fresh Torch-2.13 native builds, manifests, source/ABI/library checks and actual compiler-failure preservation pass |
| B — loader/tensor correctness | PASS for final frozen runtime | 409/409 reconstruction passes on both ABIs; full old/target V2 serving audits pass all 409 modules including 8 MTP; head/QKV/GDN loader checks and weakref ownership test pass |
| C — kernel/shape correctness | PASS for frozen production graph scope | Seven linear classes, including the full 248,320-row head, pass 18 row counts from 1–512 with unchanged RMS 2e-3, padded-tail poisoning and large-first compilation. SmallM graphs through 128 pass; INT8-prefill 129/256/512 remains eager-only and outside FULL_DECODE_ONLY max 64. [Shape receipts](../benchmarks/results/exl3-migration/optimized-shape-mixed-v1/README.md) |
| D — attention correctness | PASS for final frozen runtime | Native target attention survives API/long-context qualification; Rebuilt shared-KV verification micro gates pass C1/C4 eager/graphs, exact pages, causal poisoning, FP8 scales and native unsupported-q fallbacks. Actual M04 graph serving improves103K C1 by~14% with bit-identical512-token outputs. Unified exact-K mixed dispatch passes31 numerical XPU cases, two graph/mutated-metadata checks and three causal-poison controls; final packaged micro gates and eight mixed ABBA cases/24requests pass: cold49K TTFT~36→27s and overlapping SSE p95gap~3.0→2.05s, zero preemptions/cachehits |
| E — quality | PASS finite-panel engineering qualification; final coding review pending | Final matched native/optimized short controls have all 272 arrays bit-identical: PPL 3.646722 / KL 0.0324807. Final long suffix PPL 1.208579 / KL 0.00117770, 32/32 sampled top1 agreement. Twenty generated requests / 2,560 tokens per arm preserve argmax selection but nine histories differ; recorded margins do not justify a blanket tiny-tie claim. Independent attention tests retain original tolerance. [Quality scope and receipts](../benchmarks/results/exl3-migration/optimized-quality-v1/README.md) |
| F — capacity/stability | PASS with accepted C16 pressure | Old and target runtimes pass fresh 103K/139K/188K/200704-total, C4 4x32K, image119K and prefix/confirmed abort/restart with zero traced preemptions/rejections; matched greedy response identical. Expanded .965/262K/C16/32-image/4-video capacity passes with accepted C16 pressure: all16 complete, four events/two requests and16000 extra submitted-prefill tokens; one long image and confirmed abort/restart pass. Mixed/p95 and API/media-limit ABBA complete. Final immutable runtime completes all nine operation cases:261120+1024 exact262K boundary, C4×32K clean, all16×8K complete with four explained pool-pressure events/two requests,32 unique max-area images131K/no-preemption, confirmed abort/recovery and independent restart pass |
| G — performance | Engineering measurements complete; final public matrix pending | Controlled compact MTP3/4 screen: 36 waves / 84 requests, retain MTP3; full-vocabulary ablation: 60 requests, retain 65,536-row draft pruning. Final repeated whole-pipeline ABBA: 103K C1 43.98→49.24 tok/s cold and 43.91→49.24 warm (~12%), matched cache residency. Histories differ, so this is end-to-end evidence. Mixed ABBA improves 49K TTFT ~36→27 s and overlapping SSE gap p95 ~3.0→2.05 s. Brief final profile complete; captured graph internals remain opaque. Final 70-wave C1–C4 matrix and corrected coding pair are running. [Performance receipts](../benchmarks/results/exl3-migration/optimized-performance-v1/README.md) |
| H — rollback | Final drill pending | Independent GPTQ image, policy, launcher and caches are preserved. The final controller will load the frozen GPTQ image and run the corrected paired coding task as the actual rollback exercise. |

See [port notes](EXL3_PORT_NOTES.md) and
[the implementation plan](EXL3_PRODUCTION_MIGRATION_AND_OPTIMIZATION_PLAN.md)
for dependencies and acceptance criteria. Historical performance numbers are
not labeled as migrated-engine measurements. The user accepts preemption under genuine C16 capacity pressure and allows
C4/C8 qualification; this does not accept unmeasured throughput/quality
regressions or failed completion/recovery. See the latest port-note steering.

User operational policy: leave GPTQ offline throughout EXL3 development; start only for selected GPTQ comparisons and the final explicit rollback test. The pinned production artifact remains unchanged.

## Authorized final overnight scope

The user authorizes finishing qualification, publishing the complete README
benchmark, deploying the EXL3 profile if it passes, and committing/pushing the
completed release. Kernel profiling is limited to a brief final check; avoid
prolonged speculative tuning. Correctness and quality have priority. The final
README scope is the existing20 scenarios/70 measured waves/124 requests plus
corrected Flappy Bird v7 (10K context bands, actual coverage and functional
grading) and the independent QueueKit v2 task. This authorization does not
turn an incomplete or failing qualification into a production pass. GPTQ stays
offline through engineering apart from selected comparisons/rollback.

Final runtime image6cf48999 is frozen; release metadata child c09015ce has the
same RootFS. CPU production-launch preflight verifies native/M04/oneDNN/profile
artifacts and all11 pinned checkpoint file hashes. Neither canonical production
configuration nor production tag has been replaced. The final source snapshot
and upstream patch are published under `engine/exl3xpu`.

## Phase disposition before the final public run

| Plan phase | Disposition |
|---|---|
|0 Baseline/pins |Complete, historical images/checkpoints/API arguments/power and source receipts preserved|
|1 Build/validation |Complete, both ABI builds and fail-closed artifact/loader/numerical checks|
|2 Loader semantics |Complete,409/409 reconstruction and8/8 actual MTP serving inventory|
|3 Serving contract |Complete, existing200704/C4 contract then user-authorized expanded262144/.965/16 admission/32images/4videos|
|4 Preemption diagnosis |Complete, pool/group accounting distinguishes true C16 pressure from clean C1/C4;16000 extra submitted prefill rows reported|
|5 Runtime port |Complete, vLLM0.30/Torch2.13/oneAPI2026.1.1 and source-guarded V2 integration|
|6 Precision-path quality |Complete finite-panel regression checks; natural short/engineered long/generated scope and limitations explicit|
|7 MTP3/4 |Complete compact controlled study per user steering; MTP3 retained, full-vocabulary ablation separately complete|
|8 Shared-KV/M04 |Complete rebuilt library, original-tolerance numerical/graph/causal tests and actual serving|
|9 Guarded mixed attention |Complete numerical exact-page/mixed tests and real overlapping-prefill ABBA tails|
|10 Host scalar |Complete cached host constant and thread/stream-owned oneDNN partition fix|
|11 Native row dispatch |Complete tests retained EXL3 implementation; full head and supported capture/large-first scope verified|
|12 Kernel tuning |Brief final profile complete per user instruction; no speculative rewrite. Captured body graph internals remain opaque to the bounded trace|
|Final measured evaluation/release |Running70-wave frozen matrix, QueueKitv2, isolated prefix and corrected pairedFlappyv7/rollback; deployment/README/push pending results|

The final whole-attention-pipeline ABBA independently repeats103K C1 cold/warm:
native43.98/43.91→optimized49.24/49.24 tok/s (~12%). Cache reuse and output
budgets match; both variants are stable across repeats. Output histories differ
from token8 between variants, so this is end-to-end serving evidence, not an
isolated kernel attribution or bit-exact-text claim. See the compact
`optimized-performance-v1` evidence.

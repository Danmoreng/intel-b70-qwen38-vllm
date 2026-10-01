# EXL3 release qualification — in progress

Production decision: **do not migrate yet**. GPTQ remains pinned. This is a
progress ledger, not a completed qualification or a recommendation to release.

| Gate | Status | Evidence / remaining work |
|---|---|---|
| A — build/artifact integrity | PASS for final frozen runtime | Pinned Torch-2.12 and fresh Torch-2.13 native builds, manifests, source/ABI/library checks and actual compiler-failure preservation pass |
| B — loader/tensor correctness | PASS for final frozen runtime | 409/409 reconstruction passes on both ABIs; full old/target V2 serving audits pass all 409 modules including 8 MTP; head/QKV/GDN loader checks and weakref ownership test pass |
| C — kernel/shape correctness | PASS for frozen production graph scope | Seven linear classes including full248320-row head pass18 row counts1–512, unchanged RMS2e-3, padded-tail poisoning, compiled large-first ordering and mutated graphs. SmallM graph checks through128; large INT8-prefill129/256/512 eager-only, outside FULL_DECODE_ONLYmax64 |
| D — attention correctness | PASS for final frozen runtime | Native target attention survives API/long-context qualification; Rebuilt shared-KV verification micro gates pass C1/C4 eager/graphs, exact pages, causal poisoning, FP8 scales and native unsupported-q fallbacks. Actual M04 graph serving improves103K C1 by~14% with bit-identical512-token outputs. Unified exact-K mixed dispatch passes31 numerical XPU cases, two graph/mutated-metadata checks and three causal-poison controls; final packaged micro gates and eight mixed ABBA cases/24requests pass: cold49K TTFT~36→27s and overlapping SSE p95gap~3.0→2.05s, zero preemptions/cachehits |
| E — quality | PASS finite-panel engineering qualification | All six short-panel arms complete incl expanded full candidate, PPL3.647854/KL0.032727 versus GPTQ3.801927/0.086369. Four untruncated32K/100K/180K/262K prefix suffix probes complete, PPL1.209323 versus BF161.210748 and KL0.0015101; all32 sampled top1 values agree and native/API alignment passes. Engineered-window/reference-chunk limitations retained. Bounded generated-route diagnostics complete but show ranking flips; Final matched native/optimized short control has all272 arrays bit-identical (PPL3.646722/KL0.0324807); final long suffix PPL1.208579/KL0.00117770 with32/32 top1 agreement. Generated20requests/2560tokens perarm: all optimized choices equal a reported logits maximum,9 histories differ (margins recorded; no blanket tiny-tie claim). Independent attention numerical gates retain original tolerance. Paired functional coding remains part of final release review |
| F — capacity/stability | Partial | Old and target runtimes pass fresh 103K/139K/188K/200704-total, C4 4x32K, image119K and prefix/confirmed abort/restart with zero traced preemptions/rejections; matched greedy response identical. Expanded .965/262K/C16/32-image/4-video capacity passes with accepted C16 pressure: all16 complete, four events/two requests and16000 extra submitted-prefill tokens; one long image and confirmed abort/restart pass. Mixed/p95 and API/media-limit ABBA complete. Final independent32 maximum-area images, expanded-boundary/abort/restart and clean-pressure tests are running |
| G — performance | Partial | First4K pilot complete. Broad3-4-4-3 matrix curtailed at user request after31 completed MTP3 waves. The36-wave compact screen is complete:18 exact reused MTP3 waves plus18 fresh MTP4 waves,84 requests, zero preemptions. At103K code C1, MTP4 is only~1.1% faster;4K prose C4 regresses~8%. Warm48K C4 has unequal cache residency (~94% vs47%), so it does not establish a kernel regression. This is single-pass screening, not ABBA qualification. Both bounded component/row arms are complete (six128-token event waves plus four8-cycle traces); captured graph decomposition remains unresolved. A separate eager103K C1 trace completes and attributes attention at~40.6% of target-body summed kernel time; it is not serving-graph timing. Bounded generated diagnostics complete: all10240 main choices are reported argmax, but route-dependent ranking flips remain, so generated numerical quality stays partial. MTP3 selected as the target-profile default; full-vocabulary ablation complete (60 requests; retain65536-row pruning); M04 micro gates and eight-wave graph serving screen complete:103K C1 improves~14% (44→50 tok/s) with bit-identical512-token outputs; warm49K C4 improves~5.9% with equal cache reuse. Short effects are mixed; single-pass confidence remains limited. Matched final serving and corrected paired coding tasks remain |
| H — rollback | Partial | Exclusive native test campaign restored original GPTQ image and health; final candidate rollback drill remains |

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

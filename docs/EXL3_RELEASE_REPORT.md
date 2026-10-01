# EXL3 release qualification — in progress

Production decision: **do not migrate yet**. GPTQ remains pinned. This is a
progress ledger, not a completed qualification or a recommendation to release.

| Gate | Status | Evidence / remaining work |
|---|---|---|
| A — build/artifact integrity | PASS for tested port artifact | Pinned Torch-2.12 and fresh Torch-2.13 native builds, manifests, source/ABI/library checks and actual compiler-failure preservation pass |
| B — loader/tensor correctness | PASS for tested port | 409/409 reconstruction passes on both ABIs; full old/target V2 serving audits pass all 409 modules including 8 MTP; head/QKV/GDN loader checks and weakref ownership test pass |
| C — kernel/shape correctness | Partial | Both-ABI native and INT8 reference gates plus semantic GDN CPU/XPU/graph checks pass; expanded linear row/crossover/graph matrix remains |
| D — attention correctness | Pending | Native target attention survives API/long-context qualification; numerical attention, mixed traffic, portable verify and optimized dispatch/fallback checks remain |
| E — quality | Running | Frozen BF16 reference reused; staged short-panel target FP16/FP8/INT8/graph/MTP arms running. Full candidate precision and long suffix probes remain |
| F — capacity/stability | Partial | Old and target runtimes pass fresh 103K/139K/188K/200704-total, C4 4x32K, image119K and prefix/confirmed abort/restart with zero traced preemptions/rejections; matched greedy response identical. User-preferred .965/262K/C16/32-image/4-video profile remains unqualified |
| G — performance | Pending | Instrumented MTP3/MTP4, matched serving matrix and corrected paired coding tasks remain |
| H — rollback | Partial | Exclusive native test campaign restored original GPTQ image and health; final candidate rollback drill remains |

See [port notes](EXL3_PORT_NOTES.md) and
[the implementation plan](EXL3_PRODUCTION_MIGRATION_AND_OPTIMIZATION_PLAN.md)
for dependencies and acceptance criteria. Historical performance numbers are
not labeled as migrated-engine measurements. No remaining regression has been
accepted by the user.

User operational policy: leave GPTQ offline throughout EXL3 development; start only for selected GPTQ comparisons and the final explicit rollback test. The pinned production artifact remains unchanged.

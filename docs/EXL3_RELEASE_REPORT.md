# EXL3 release qualification — in progress

Production decision: **do not migrate yet**. GPTQ remains pinned. This is a
progress ledger, not a completed qualification or a recommendation to release.

| Gate | Status | Evidence / remaining work |
|---|---|---|
| A — build/artifact integrity | Partial | Baseline and Torch-2.12 safe build/failure injection pass; target Torch-2.13 build remains |
| B — loader/tensor correctness | Partial | Shared index covers 409 modules, all native reconstruction checks pass; old runtime complete serving audit passes 409/409 including 8 MTP; target-runtime audit remains |
| C — kernel/shape correctness | Partial | Original decode paths and INT8 reference gate pass; expanded rows, crossover and graph matrix remains |
| D — attention correctness | Pending | Target-runtime attention, mixed traffic and fallback checks remain |
| E — quality | Pending | Historical BF16/quantization evidence preserved; migrated production-precision arms remain |
| F — capacity/stability | Partial | Old-runtime targeted GDN retirement fix passes fresh 103K/139K/188K/200704-total with zero preemptions and identical 22K greedy response; old-runtime long C4/image/prefix-abort also pass; all target-runtime gates remain |
| G — performance | Pending | Instrumented MTP3/MTP4, matched serving matrix and corrected paired coding tasks remain |
| H — rollback | Partial | Exclusive native test campaign restored original GPTQ image and health; final candidate rollback drill remains |

See [port notes](EXL3_PORT_NOTES.md) and
[the implementation plan](EXL3_PRODUCTION_MIGRATION_AND_OPTIMIZATION_PLAN.md)
for dependencies and acceptance criteria. Historical performance numbers are
not labeled as migrated-engine measurements. No remaining regression has been
accepted by the user.

User operational policy: leave GPTQ offline throughout EXL3 development; start only for selected GPTQ comparisons and the final explicit rollback test. The pinned production artifact remains unchanged.

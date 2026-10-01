# EXL3 release qualification — in progress

Production decision: **do not migrate yet**. GPTQ remains pinned. This is a
progress ledger, not a completed qualification or a recommendation to release.

| Gate | Status | Evidence / remaining work |
|---|---|---|
| A — build/artifact integrity | Partial | Baseline and Torch-2.12 safe build/failure injection pass; target Torch-2.13 build remains |
| B — loader/tensor correctness | Partial | Shared index covers 409 modules, all native reconstruction checks pass; complete serving loader audit remains |
| C — kernel/shape correctness | Partial | Original decode paths and INT8 reference gate pass; expanded rows, crossover and graph matrix remains |
| D — attention correctness | Pending | Target-runtime attention, mixed traffic and fallback checks remain |
| E — quality | Pending | Historical BF16/quantization evidence preserved; migrated production-precision arms remain |
| F — capacity/stability | Pending | Equal-contract fresh-worker long-context, C4, abort, prefix and image cases remain |
| G — performance | Pending | Instrumented MTP3/MTP4, matched serving matrix and corrected paired coding tasks remain |
| H — rollback | Partial | Exclusive native test campaign restored original GPTQ image and health; final candidate rollback drill remains |

See [port notes](EXL3_PORT_NOTES.md) and
[the implementation plan](EXL3_PRODUCTION_MIGRATION_AND_OPTIMIZATION_PLAN.md)
for dependencies and acceptance criteria. Historical performance numbers are
not labeled as migrated-engine measurements. No remaining regression has been
accepted by the user.

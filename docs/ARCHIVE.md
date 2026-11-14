# Historical reports and experiments

[Back to README](../README.md)

These documents preserve the evidence, failed checks and decisions that led to
the current engine. Their statements about "current", "production" or pending
work apply to their own date and image. For today's supported settings and
rollback target, use [configuration](CURRENT_CONFIGURATION.md),
[operations](OPERATIONS.md) and [current benchmarks](BENCHMARKS.md).

## Release evidence

| Date / topic | Preserved report |
|---|---|
| 2026-11-12: current profile qualification | [1792/128 cache, COW correction and deployment](../benchmarks/results/mtp-cache-1792-20261112/README.md) |
| 2026-11-10: previous matrix profile | [MB4/K4 serving comparison and deferred investigations](../benchmarks/results/mtp-matrix-20261110/README.md) |
| 2026-11-08: runtime upgrade | [vLLM 0.33 qualification](../benchmarks/results/vllm-033-upgrade-20261108/README.md) |
| 2026-11-08: RAM layout | [LBNHC comparison, measured at 230 W](../benchmarks/results/ram-kv-layout-20261108/README.md) |
| 2026-10-10: earlier stack | [oneDNN/Torch qualification and adverse numerical results](../benchmarks/results/b70-stack-20261010/FINAL_DECISION.md) |
| Original EXL3 release | [v1 report](EXL3_RELEASE_REPORT.md) and [v2 report](EXL3_RELEASE_V2_REPORT.md) |

The earlier stack report retains failed bitidentity/equality receipts and the
limited PPL/KL scope. Passing later finite checks does not retroactively turn
those failures into passes.

## Historical benchmarks and experiments

- [EXL3 v2 power-limit comparison](EXL3_POWER_COMPARISON.md): 150/180/230/275 W,
  vLLM 0.30 / Torch 2.13. The 180 W cap remains the production choice; the old
  comparison does not establish an optimum for the current stack.
- [Flappy Bird and QueueKit](EXL3_CODING_BENCHMARKS.md): historical v1 coding
  results, including QueueKit's failed task.
- [GPTQ/EXL3 comparison](EXL3_GPTQ_COMPARISON.md): the specific older checkpoints
  and precision paths, not a general claim about all quantizers.
- [MTP4 and quantized DFlash decision](../benchmarks/results/dflash2-pro-optimization-v031/DECISION.md):
  completed experiments; production retains MTP3.
- [RAM admission candidate](KV_RAM_ADMISSION_CANDIDATE.md) and
  [exclusive-tiering design](KV_RAM_EXCLUSIVE_TIERING.md): candidate/design
  scopes. The current cache is inclusive, not an exclusive two-history tier.

## Plans and prior documentation

- [2026-10-01 migration and optimization plan](EXL3_PRODUCTION_MIGRATION_AND_OPTIMIZATION_PLAN.md):
  a historical plan written when GPTQ was production and EXL3 a candidate.
- [Earlier GPTQ/oneDNN release checklist](production-release-checklist.md) and
  [scorecard](production-release-scorecard.md): historical gates and pending items.
- [Port notes](EXL3_PORT_NOTES.md) and [Pro review follow-up](EXL3_PRO_REVIEW_FOLLOWUP.md).
- [README before the 2026-11-14 consolidation](archive/README-before-consolidation-20261114.md):
  preserves the former narrative, including known stale statements.

## Saved releases

[config/releases](../config/releases) retains images/policies, source snapshots
and runtime modules where provided. Older releases are linked for deliberate
reproduction, not selected as today's automatic rollback:
[v6](../config/releases/exl3-v6/README.md),
[v5](../config/releases/exl3-v5/README.md),
[v4](../config/releases/exl3-v4/README.md),
[v3](../config/releases/exl3-v3/README.md),
[v2](../config/releases/exl3-v2),
[v1](../config/releases/exl3-v1/README.md) and
[GPTQ](../config/releases/gptq-onednn-v2/README.md).

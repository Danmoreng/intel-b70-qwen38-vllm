# Fixed-MTP3 stack and kernel campaign

The user supplied `B70_MTP3_CODEX_IMPLEMENTATION_PLAN_EN.md` and authorized
overnight execution on 2026-10-10. The source plan is retained next to this
file. Production defaults are immutable while experiments run.

First commit scope: diagnostic and experiment infrastructure only. `inventory.py`
inspects the actual loaded worker, `profile_decode.py` uses the pinned-image
runner event/trace adapter, `analyze_profile.py` reports disjoint/ambiguous costs,
and `gates.json` freezes decision criteria before candidate measurements.

P0 starts with 64K C1 and frozen 32K prose C4. Event and kernel-trace waves are
diagnostics, never throughput evidence. Instrument-free controls use the same
frozen IDs/seeds. An isolated worker holds the existing exclusive GPU lock;
the pinned production service is restored in a finally block.

Baseline: image 194a8786..., MTP3, FP16, FP8 KV, 262144 total tokens, 180W,
max-seqs16, max-batched-tokens4096, memory .965 and FULL_DECODE_ONLY graphs.
All dependency/kernel candidates are conditional on the measured route/cost
report and are separate logical changes. The exact plan defines stop rules.

Raw receipts live under `benchmark-results/b70-stack-20261010/`; compact reviewed
results are published under `benchmarks/results/b70-stack-20261010/` later.
Existing references remain in the original qualified checkout and are reused,
not regenerated. Historical failed MB4/full-fusion/async/piecewise experiments
remain negative evidence unless a new, falsifiable hypothesis justifies retry.

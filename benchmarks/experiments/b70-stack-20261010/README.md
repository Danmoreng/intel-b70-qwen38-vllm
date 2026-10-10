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

`run_d64.py` captures bounded actual same-prefix operands on the already qualified v0.31 image and, only for named missing references, the archived v0.30 image. `compare_d64.py` independently checks actual greedy-Dirac rejection against the scalar Philox known-answer oracle. Raw logits remain private. `run_quality.py` captures only the changed-stack candidate and compares against stored current-v0.31 arrays, then reuses the guarded state replay; a failed exact comparison stays failed pending numerical review.

`serving_screen.py --protected` adds the existing 4K/128K code fixtures and the exact source32K fixture at C4. `analyze_screen.py` preserves all independent blocks, including first-touch compiler penalties; additional blocks do not replace unfavorable earlier observations. `run_operations.py` exercises only the candidate, including real decode/prefill overlap, media, abort, prefix extension and admission pressure. All GPU controllers restore the pinned qualified image in finally.

`freeze_release.py` prepares candidate metadata only. It cannot approve or deploy a release; remaining gates and the final frozen serving campaign are required.

The exact final image's first operational attempt stopped on a reporting path
error (now fixed); its second completed C16 but recorded three preemptions
against the archived count of two. Both failed receipts remain immutable.
`compare_c16.py` supplies the specifically missing contemporaneous reference
with three balanced, fresh-worker baseline/candidate pairs and identical
API/media/C4 antecedents. It does not regenerate the old quality or full
serving benchmark. A separately checksummed engineering review is required
before one bounded final-operations retry. The retry retains the original
maximum of two preemptions and zero recomputation; there is no retry-until-pass
loop and no increased numerical tolerance or admission/memory setting.

The upstream ordering-fix-only fallback is prepared as a read-only patch
applicability review under `candidates/E01-minimal/`. No fallback library was
built or tested. It is conditional on a clear unrelated stable-library
regression, not another mandatory dependency sweep.

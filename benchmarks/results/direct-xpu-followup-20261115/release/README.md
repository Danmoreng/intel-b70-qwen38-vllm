# EXL3 v8 production release — 2026-11-15

The unchanged, fully tested R12 image is approved as the new production
profile. The permanent-service deployment and restart are recorded in
[deployment/promotion.json](deployment/promotion.json).

| Identity | Value |
| --- | --- |
| Image tag | `local/b70-qwen38-vllm:production-exl3-v8` |
| Exact image | `sha256:131dd9e11809de2f3136547618f130ee238f0b90cd5718567bcdd0dad5360e19` |
| Runtime | vLLM 0.33.0, official Torch 2.15.0+xpu |
| Policy SHA256 | `2d232cb1344ea5faf358937b0c4e641cc0647423f6a3385d9213f73bd259db48` |
| Immediate rollback | [Saved v7](../../../../config/releases/exl3-v7/README.md), `dc818253…` |

The direct official XPU base replaces the old local upgrade chain. The engine
retains standard model compilation, MTP3, native decode graphs, 1792/128 cache
geometry, 16 slots, 12 GiB RAM and 180 W. The official Torch wheel is intact.
The image carries the upstream Intel PTI graph-lifetime correction and the
minimal packed M04 empty-split fix. No diagnostic instrumentation is installed.

## Review and evidence

- [Explicit operator acceptance](operator-acceptance.json), after the full
  matrix and targeted regression follow-up.
- [Decision](decision.json), [all final-image gates](gate-assessment.json),
  [benchmark summary](benchmark-summary.json) and
  [review of every performance trigger](performance-review.json).
- [Full qualification](../../direct-xpu-20261114/README.md): all nine stages
  pass, aligned finite quality windows, natural coding tasks 4/4, 70 measured
  waves / 124 requests with exact paired inputs and zero preemptions.
- [Targeted follow-up](../README.md): 38 measured waves / 50 requests,
  fresh and after-128K conditions; the large 2K/C2 stalls do not recur in
  six new-image repeats. This does not exclude rare recurrence.
- [132 runtime artifact hashes](runtime-artifacts.json),
  [frozen image profile](profile.json) and
  [deployment metadata explanations](release-metadata-notes.json).
- [Read-only v7 rollback preflight](rollback-preflight.json) passed after the
  permanent-service restart; no old-engine restoration was performed.

The full-matrix summed wave time is **1.82% longer** than v7. The additional
4K/C1 sample retains **+6.07% median time / +2.73% summed time**, consistent
with lower MTP acceptance and nearly unchanged estimated round cost. These
costs are explicitly accepted; the original regressions are not rewritten as
passes without review. The open sporadic BCS GPU fault is not claimed fixed.

Historical receipts under the qualification and follow-up directories keep
their original “not approved” statuses. This later decision supersedes their
administrative approval state; their measurements and failures remain intact.
The deployment binds the unchanged tested RAM modules to the exact R12 image,
replacing the unused R11 draft manifest only for deployment. The frozen policy
keeps its provenance; the approved decision and current release manifest name
the actual preceding v7 as immediate rollback.

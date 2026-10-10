# B70 MTP3 stack campaign — evidence index

**Maintenance survivor deployed, restart-verified and published. No performance survivor.**

The supplied [implementation plan](../../experiments/b70-stack-20261010/B70_MTP3_CODEX_IMPLEMENTATION_PLAN_EN.md)
and [pre-registered gates](../../experiments/b70-stack-20261010/gates.json)
define scope. Baseline is the qualified v0.31 MTP3 image `194a8786`:
FP16 target/draft, FP8 KV, 262,144 total tokens, 16 admitted sequences and
180 W. This campaign does not reopen DFlash, MTP-depth, power-limit or
whole-host upgrade experiments.

| Question | Evidence |
| --- | --- |
| Actual routes, graph shapes and component budgets | [P0 report](P0-route-and-cost-report.md) |
| Bounded old/new same-prefix state and rejection diagnosis | [D64 review](D64-same-prefix-review.md) |
| Dependency/kernel decisions | [E01](decisions/E01.md), [E02](decisions/E02.md), [E03](decisions/E03.md), [E04a](decisions/E04a.md), [E04b](decisions/E04b.md), [E05](decisions/E05.md), [E06](decisions/E06.md), [E07](decisions/E07.md) |
| Exact old/stable build and loaded artifacts | [Build receipts](E01-build-public-receipts.json), [local gates](E01-build-and-local-gates.json) |
| Numerical changes and retained exact-equality failure | [Changed-stack quality](E01-changed-stack-quality.json), [explicit review](E01-numerical-review.json) |
| Difficult attention inputs, real linear weights and cache pressure | [SDPA edges](E01-sdpa-edge-values.json), [linear references](E01-linear-reference.json), [in-flight pressure](E01-inflight-cache-pressure.json) |
| Actual usable boundary and matched KV pools | [Memory ledger](E01-memory-ledger.json), [first-touch accounting](E01-firsttouch-accounting.json) |
| Independent startup-paired performance screen | [Eight core / five protected blocks](E01-confirmation-eight-blocks.json), [counter decomposition](E01-eight-block-counter-decomposition.json) |
| Failed operational attempts and missing contemporary C16 reference | [Original attempts](E01-operations-attempts.json), [matched counters](E01-c16-matched-counters.json), [review](E01-c16-matched-review.json) |
| Exact frozen candidate's strict preflight and operations | [Preflight](E01-final-strict-preflight.json), [operations](operations.json), [assessment](assessment.json) |
| Complete final frozen serving matrix | [Summary](serving-summary.json), [all historical comparison cells](benchmark-comparison.json), [original review-trigger checkpoint](final-review-pending.json) |
| Exact-fixture historical-trigger controls | [All paired/per-fixture results](E01-frozen-trigger-reference-report.json), [observed clocks and thermal coverage](E01-frozen-trigger-comparability.json), [exact public export map](E01-frozen-reference-public-exports.json) |
| Actual installed CPU model-route guard | [Passing CPU-only check](evidence/E01-cpu-route-guard-r2/report.json) |
| Final candidate/release decision | [Decision report](FINAL_DECISION.md), [explicit performance tradeoff](performance-review.json), [promotion approval](decision.json) |
| Publication and requirement-by-requirement completion | [Manifest](PUBLICATION_MANIFEST.json), [published payload](PUBLICATION_RECEIPT.json), [audit](COMPLETION_AUDIT.json) |

The final matrix completed 70 waves / 124 requests plus three isolated 64K
prefix resends. Its historical adverse cells remain visible. Three balanced
fresh-worker control pairs completed only those named missing
contemporary references, with every original prompt, request value, repeat
and 1,024-output budget retained. The final decision permits a maintenance
promotion with smaller systematic C4 loss explicitly retained; no weighted
performance gain is established. Actual permanent-service start/restart passed; [sanitized receipt](promotion.json).

Public JSON includes scalar counters, timings, output hashes and source
identities. Private raw prompts, responses, streams and full-vocabulary
arrays remain in local `benchmark-results/b70-stack-20261010/`. Licensed
runtime source snapshots are included as source code, not model weights or
native binaries. An intermediate build's `GPU_UNTESTED` status and a failed
exact-array/operational receipt are never rewritten as a later pass.

Recorded trace/event measurements describe routes and finite component
budgets; their sums are not additive critical-path percentages. Throughput
controls have profiling hooks off. Small paired-bootstrap intervals do not
constitute universal equivalence or population-tail guarantees. Historical
power comparisons remain at their older qualified stack; only fixed-cap
comparability is observed here.

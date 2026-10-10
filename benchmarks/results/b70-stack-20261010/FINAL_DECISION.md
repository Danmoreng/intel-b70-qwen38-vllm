# Final campaign decision — B70 MTP3 stack

**MAINTENANCE_SURVIVOR: oneDNN 3.13.4 selected for promotion. No performance survivor.**

The reviewed [release decision](decision.json) and
[performance tradeoff](performance-review.json) select frozen image
`36e41ea027758060f9352bdd7f9ed47de5143d0424b238653960dbf8357c176a`,
policy `d2b7a88f5c8646f8626c3d4f86e96acac0f476be9dbaa1eb6424063572364649`.
Actual permanent-service deployment/restart is **pending** at this decision
checkpoint. The old qualified image `194a8786` remains running until that
separate deployment succeeds; the [saved v3 release](../../../config/releases/exl3-v3/README.md)
and strict rollback launcher are retained.

The maintenance benefit is the identified upstream SDPA initialization/barrier
ordering correction, absent in the audited old source/mapped binary and
present in stable 3.13.4. The existing serving race was not reproduced, so
this is a relevant preventive fix rather than a claimed repair of a measured
production failure. The matching EXL3 rebuild retains source `fc8f8c9f`,
compiler/toolchain/ABI flags, original M04, vLLM 0.31.0, Torch 2.14.0+xpu
and Triton 3.8.0+xpu. No model recipe, scheduler, sampling or MTP-depth
change is selected.

| Candidate | Final disposition | Basis |
| --- | --- | --- |
| [E01](decisions/E01.md) — stable oneDNN | MAINTENANCE_SURVIVOR | Relevant upstream ordering fix; progressive gates and final panel pass with explicit numerical/performance reviews |
| E01-minimal — old SDK plus ordering fix | Not required, not built | Patch applies; no clear unrelated stable-library regression was confirmed |
| [E02](decisions/E02.md) — GDN norm | NO_MATERIAL_GAIN, not implemented | Actual compiled norm budget is about 0.210/0.247 ms in C1/C4 |
| [E03](decisions/E03.md) — fused QKV replacement | NO_MATERIAL_GAIN, not implemented | Existing fused route is about 0.054/0.060 ms |
| [E04a](decisions/E04a.md) — M04 reduction tiling | NO_MATERIAL_GAIN, implemented operator prototype | Forty original-tolerance/bitwise and mutation cases pass; whole-call q4 long-context/C4 improvement is not material |
| [E04b](decisions/E04b.md) — native-layout output | NO_MATERIAL_GAIN, not implemented | All traced copies together are small; no material M04 epilogue copy isolated |
| [E05](decisions/E05.md) — norm/HadIn preparation | NO_MATERIAL_GAIN, not implemented | Small estimated eligible budget; preserve both Half rounding boundaries |
| [E06](decisions/E06.md) — scheduling/graph pipeline | INCONCLUSIVE, not implemented | No identified avoidable host operation; uncovered trace intervals are not proven removable idle |
| [E07](decisions/E07.md) — Torch 2.14.1 | INCONCLUSIVE, held before execution | No identified relevant XPU fix and installed vLLM's exact Torch pin unresolved; no force-install |

No individually demonstrated performance winners remain to combine. The
original M04 is selected. These finite measurements do not prove a globally
optimal kernel for every workload; they support stopping this conditional
campaign instead of reopening previously rejected broad fusion/async changes,
DFlash, MTP4+, BF16 execution, lower context or whole-host upgrades.

## Performance and explicit tradeoffs

Eight independent core startup-paired blocks plus five protected blocks
yield weighted cold decode +0.111% (descriptive 95% interval −0.701% to
+0.893%) and warm −0.752% (−1.865% to +0.236%). Neither meets the
pre-registered ≥3% weighted speed promotion rule. Warm prose-C4 client rate
is −2.464% (−5.363% to +0.132%); its adverse result and broad uncertainty
are retained, not characterized as a tight equivalence proof.

The complete frozen matrix is 70 waves / 124 requests plus three isolated
64K prefix resends. The [historical comparison](benchmark-comparison.json)
records 4K C1 native decode −4.707%, 32K C1 −3.199%, and 4K C4 median TTFT
+6.356%. All historical cells remain visible.

Three balanced fresh-worker pairs for exactly those missing contemporary
references retain every original repeat, byte-identical prompt, request
value, warmup and 1,024-output budget. Their paired per-fixture effects are:

| Protected observation | Mean paired effect | Descriptive 95% startup-block interval |
| --- | ---: | ---: |
| 4K C1 native decode | +1.873% | −2.160% to +4.211% |
| 32K C1 native decode | +1.404% | +0.581% to +2.945% |
| 4K C4 client decode rate | −0.923% | −1.413% to −0.152% |
| 4K C4 median TTFT | −0.906% | −2.926% to +0.189% |
| 4K C4 finite-panel request E2E p95 | +0.882% | +0.122% to +1.544% |

The smaller consistent C4 loss is an explicit maintenance tradeoff. Alternate
within-arm median ratios show −1.321% C4 client rate; both conventions and
every per-fixture effect remain in the [control report](E01-frozen-trigger-reference-report.json).
No clear threshold-exceeding loss is confirmed in the protected case or
per-fixture intervals. Three paired starts are low-powered, not universal
noninferiority or population-tail proof. Eight overlapping-counter windows
retain a one-token round-accounting residual and remain invalid; their
unfiltered counter rates are observations, not validated physical round
timings. All actual output counts are exact. C4 overlapping windows are valid.

[Observed comparability](E01-frozen-trigger-comparability.json) retains 188
15-second snapshots at the fixed 180-W cap with no sampled thermal flag.
The first baseline arm has incomplete early coverage, which remains an
explicit limitation. No power-limit matrix was repeated. Changed output
histories/acceptance and request-round counter decomposition do not prove
the cause of every speed difference or make losing client data irrelevant.

## Correctness, capacity and operational evidence

The exact short-array comparison failed and remains failed. Original-tolerance
independent operator/reference, difficult-input, real-weight, live-buffer
graph and actual-state replay checks pass. The separately hashed numerical
review accepts the finite quality panel: short PPL 3.642167 versus stored
3.642175, mixed KL/JS changes, and 32K suffix PPL 1.183419 versus 1.190357.
This is not general accuracy improvement or free-running text identity.
Stored BF16 references are evaluated on CPU, with FP16 target and MTP runtime.

An actual 261,120-input + 1,024-output boundary request passes on the tested
runtime parent, whose RootFS is identical to the frozen final image. Matched
ordinary starts retain 200 KV blocks / 11,141,120,000 bytes; deliberately
empty compiler caches give 199 on both old/new. No theoretical block-to-token
conversion or KV override substitutes for the boundary test. The 262,144
total-token limit, 16 admitted sequences, max-batch 4,096, memory fraction
0.965, FP8 KV, cache capacity 64, full decode graphs and 180 W are preserved.
Sixteen admissions do not imply sixteen simultaneous full-context requests.

The exact final image passes API/media/C4, pressure C16, long image, abort/
prefix extension, real two/four-request prefill/decode overlap, 32 independent
4.2MP-image inputs and a 200-block restart. The final C16 pressure count is
two with zero excess recompute. Original reporting/C16 failures, archived
count two, and matched current controls (old 2/3/3, new 2/2/2) remain visible.
Only one bounded final operational retry followed the reviewed comparison.

The installed model-route predicates additionally reject otherwise-shaped
CPU inputs without custom native calls or GPU tensors. The first CPU check
failed on image file-read permissions before execution; the identical script
passes with the production image user. This verifies route exclusion, not
CPU support in the direct XPU-native API.

The bounded D64 comparison and independent native rejection-prefix/Philox
checks do not identify the cause of the prior v0.30→v0.31 64K performance
loss. That investigation remains inconclusive; no sampler/state patch or
unbounded bisect is claimed. See the [evidence index](README.md),
[exact-image assessment](assessment.json), [numerical review](E01-numerical-review.json),
[memory ledger](E01-memory-ledger.json) and [operations](operations.json).

## Deployment and publication checkpoint

Permanent-service promotion, real first start/restart receipt, qualified
source/default README refresh, public hash manifest and authorized GitHub
push are pending. The goal is not complete at this checkpoint. Historical
power tables will retain their values and older-stack scope; 180 W remains
the previously selected performance-per-watt operating point.

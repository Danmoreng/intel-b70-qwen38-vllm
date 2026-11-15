# Direct vLLM XPU qualification — 2026-11-14

**R12 was approved and deployed as EXL3 v8 on 2026-11-15.** See the
[release decision and deployment](../direct-xpu-followup-20261115/release/README.md).
The engine
builds directly on official vLLM 0.33 XPU, with the official PyTorch 2.15 wheel,
newer Intel userspace, a targeted Intel PTI backport and a correction to our
packed M04 causal attention. Standard model compilation, MTP3 and native graphs
are enabled. No diagnostic tracing is installed in the release image.

See the [build instructions](../../../engine/direct-xpu/README.md) and
[frozen image and test contract](qualification/m04-fixed-r12/frozen-contract.json).
The immutable release image is
`sha256:131dd9e11809de2f3136547618f130ee238f0b90cd5718567bcdd0dad5360e19`.

## Current final-image results

| Gate | R12 result |
| --- | --- |
| Build, imports, native ABI and source identities | Pass |
| Independent M04 boundary calculation | 40/40 cases and four changed-input graph replays pass |
| Profiled native graphs and original kernel tolerances | Pass |
| Model/state replay | Four cases pass |
| Startup and concurrent partial-page COW | Pass |
| Frozen coding tasks with independent hidden tests | 4/4 pass, natural completion |
| Teacher-forced quality | All 16 short and four long windows finite and API-aligned |
| Operations, media, RAM retention, capacity and restart | Pass |
| Complete serving comparison | 70/70 valid waves; tradeoffs reviewed and accepted |

These are results on this exact image, rather than inherited passes from an
earlier candidate. Detailed final-image receipts are archived with the
verification result. The earlier v7 coding reference scored 3/4; the original
candidate requirement remained 4/4, with no repaired solutions or selected
replacement runs. R12 completes the previously looping task in about 45 seconds.

The [quality comparison](qualification/m04-fixed-r12/quality-comparison.json)
uses the same frozen inputs as v7: short-panel perplexity changes by -0.038%,
long-suffix perplexity by -0.186%, and full long-prompt perplexity by -1.253%.
Negative values favor the candidate. Paired window uncertainty includes zero;
these finite panels do not establish a general quality improvement. No
numerical tolerances or task acceptance criteria were relaxed.

The [complete comparison](qualification/m04-fixed-r12/serving-comparison.json)
verifies all 124 payloads and prompt bytes, exact output lengths and zero
preemptions. Total measured answer time is **1.82% longer**, native weighted
decode is **0.35% faster**, and native weighted prefill is **6.53% slower**.
The [targeted follow-up](../direct-xpu-followup-20261115/README.md) and
[explicit operator acceptance](../direct-xpu-followup-20261115/release/operator-acceptance.json)
complete the performance review; the original losses remain in these receipts.

Two slow waves in `concurrency-2k-c2` account for essentially the full observed
loss: candidate times are 28.42, 20.27 and 61.85 seconds, versus 19.34, 19.99 and
21.22 seconds in v7. The remaining 67 waves together are 6.07 seconds faster.
That exclusion only localizes the issue; it does not replace the full score.
No compile-cache writes occurred during those stalls, and later Triton JIT
warnings do not prove their cause. The 4K single-request decode/wall medians
also trigger review, with overlapping observed ranges.

The subsequent follow-up below tests the exact flagged requests on unchanged
images and existing caches, both fresh and after the original 128K prologue.
No further GPU run started near the daily closing phase on November 14. The
[original verification receipt](qualification/m04-fixed-r12/verification-result.json)
and [campaign](qualification/m04-fixed-r12/gates/campaign.json) retain their
original approval state. The later operator decision accepts the measured
tradeoffs; investigation thresholds were not redefined to hide the losses.

## Targeted follow-up on 2026-11-15

[38 additional measured waves](../direct-xpu-followup-20261115/README.md) on the
same two immutable images repeat the 2K/C2 and 4K/C1 cases with fresh workers
and after the original 128K sequence. The large 2K stalls do not reproduce in
six new-image waves. The 4K median cost remains: 17.63 to 18.70 seconds (+6.07%),
with total time across ten answers +2.73%. Draft acceptance is lower while
estimated verification-round time is nearly unchanged. This follow-up does
not replace or erase the full-matrix results above. Original v7 production was
restored once after those measurements, then the operator accepted the review
and authorized the unchanged R12 production deployment.

## Corrected packed-attention regression

A trace of the actual V2 worker reproduces the original failing token sequence.
Layer observation finds the first NaN in decoder layer 3's attention output,
before the LM head and sampler. An empty causal split was normalized as `0/0`
under the newer kernel's split-normalization contract. The previous kernel
kept partial outputs unnormalized and skipped empty splits.

The minimal correction emits a zero contribution for that legal empty split.
An independent FP32 oracle fails 11/40 boundary cases without it and passes
40/40 with it. The captured real request preserves its first 885 tokens and
then completes a valid tool call. The [diagnostic report](qualification/pro-analysis-resume/README.md)
links the source comparison, patch, matched probes and first-invalid traces.
This finding supersedes the earlier, unproven Inductor hypothesis. Standard
compilation is restored; no PyTorch arithmetic patch or dependency downgrade
is used for this failure.

Historical [R10 coding failure](qualification/final/coding-failure.json),
[raw probe comparison](qualification/generation-probes/comparison.json) and
[earlier candidate campaign](qualification/final/campaign.json) remain retained.
They describe different images and do not override the final R12 results.

## Confirmed graph/profiler regression and correction

After XPU profiling, the original Torch 2.15/PTI 1.2.1 stack can replay a
later graph with incorrect output. The unchanged seven-class checkpoint
matrix fails at gate-up, two rows: relative norm **1.00021231174469** against
its fixed **0.002** limit. Eager execution passes.

A standalone reproduction also fails using only simple PyTorch operations,
without vLLM, EXL3, Triton compilation or model weights. The matching 2.14.1
control passes. The fault is not specific to the checkpoint or our kernels.

Intel's [native-graph lifetime fix](https://github.com/intel/pti-gpu/commit/dd8940ff817f94461795708fe3c386ab9e09d3b1)
releases profiler-owned graphs and invalidates stale handles when graphs are
destroyed or tracing stops. The published PTI 1.2.1 library embeds source
revision `25a77ed7693ac155bc62e563060b0976f9934b53`, which lacks that fix.

| Matched diagnostic | Observed result |
| --- | --- |
| PTI 1.2.1 rebuilt without the fix | Wrong two-row graph output after profiling |
| Same source, compiler and dependencies, with the four-file upstream fix | Full original matrix, attention and GDN checks pass |
| Fixed PTI, native graph replay under GPU profiling | Correct outputs and all nine expected GPU kernel events |

[Control receipts](pti-fix-controls.json) record image identities, source hashes
and outcomes. The [standalone regression test](../../experiments/vllm-033-upgrade/profile-graph-replay.py)
checks graph creation, destruction, mutated inputs, profiling transitions and
actual replay kernel records. Original matrix thresholds are unchanged.
The [PTI build](../../../engine/direct-xpu/pti/README.md) records the backport,
source/dependency identities and library hashes.

A temporary workaround using non-native SYCL recording passed matrix/model
math but omitted replay kernel records. It is not selected: the PTI fix keeps
native recording and kernel-level replay profiling, with no Torch override.

## Transformers compatibility exception

The full Torch2.15/PTI-fixed candidate passes real partial-page C2 forks,
RAM restoration and the API group, then fails the unchanged small-video gate
with Transformers5.23. The otherwise identical image with **5.22.2** passes
one/four videos, rejection over the count limit and oversized-image handling.
[Matched control evidence](qualification/transformers-media-compatibility.json)
retains the failed attempt and the successful control. The selected version
is the newest patch in the prior stable minor and is above the reference5.17.
Torch2.15, PTI and the newer Intel/XPU components remain selected.

A CPU-only control also exposes5.23's new validation rejecting vLLM's
second text pass after video preprocessing. DirectHF video processing alone
works in both versions. The exact cause of the inflated API token estimate is
not proven; this is a measured compatibility pin, not a relaxed media test.

## Scope and limits

Only this Linux Arc Pro B70 has been tested. The host kernel/DRM driver is
unchanged; updated Intel driver components are userspace packages inside the
image. Broader GPU impact has not been established. The
[upstream references](UPSTREAM_ISSUES.md) distinguish the PTI correction from
separate graph reports.

The user granted exclusive GPU access and requested that the old engine stay
stopped between tests. The current controller therefore stops its own test
containers without restarting v7 between stages. Daily work and shutdown
limits remain enforced.

The [pre-existing sporadic BCS fault](reference-restore-fault.json) remains open.
A historical v7 mixed-load RAM-retention miss and its passing fresh-worker
control remain recorded in [the failed measurement](qualification/baseline-mixed-load-ram-failure.json)
and [isolated repeat](qualification/baseline-isolated-ram.json). Both final arms
use the same fresh-worker ordering; arbitrary prior-load retention is not
guaranteed. Several components exceed package metadata constraints, so no clean
`pip check` is claimed. The official oneDNN 3.14.2 source tag still reports
3.14.1 internally, as recorded [separately](onednn-version.json).

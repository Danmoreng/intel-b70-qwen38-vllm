# Direct vLLM XPU qualification — 2026-11-14

The selected candidate builds directly on official vLLM 0.33 XPU with
**unmodified PyTorch 2.15**, matching Triton/XPU kernels, latest selected Intel
userspace and a targeted upstream PTI fix. See the
[build instructions](../../../engine/direct-xpu/README.md).
The earlier Torch2.15/PTI-fixed release child passed CPU integrity, strict launcher preflight,
profiler replay, the full original kernel suite, four model-state replay cases
and startup/decode. [Receipts](qualification/core/campaign.json) identify that
earlier image. Its later media gate exposed a Transformers5.23 compatibility
regression. The selected clean build uses5.22.2; its full exact-image
qualification remains pending. Production manifests still identify v7.

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

## Remaining qualification

The final direct-build image must independently pass the complete numerical
suite, model-state replay, startup/decode, frozen short/long quality panel,
operational/cache/capacity checks and the full 70-wave/124-request serving
matrix against a fresh reference run. Diagnostic success is not a substitute
for those final-image results.

The earlier [diagnostic receipts](diagnostics.json) retain version-isolation
experiments; they are not the selected configuration. The official oneDNN
3.14.2 source tag still reports library version 3.14.1, recorded
[separately](onednn-version.json). The exact source-boundary adaptation for
model replay is documented in [its receipt](replay-source-adaptation.json).

## Scope and known limits

Only this Linux Arc Pro B70 has been tested. The PTI lifetime code is not
model-specific; reproduction on other GPUs has not been established.
[Upstream references](UPSTREAM_ISSUES.md) distinguish the matching correction
from separate graph-related reports.

GPU tests run exclusively and restore the original service image. The host
kernel/DRM driver is unchanged; driver updates here mean Intel userspace
inside Docker. The [pre-existing sporadic BCS fault](reference-restore-fault.json)
remains open and is separate from the corrected arithmetic regression.
Several selected components intentionally exceed package metadata constraints;
no clean `pip check` is claimed.

## Standalone regression check

With exclusive GPU access, run the script in a locally built image. It uses no
model weights and writes numerical results plus a Chrome trace:

```bash
mkdir -p benchmark-results/pti-check
docker run --rm --network none --device /dev/dri \
  -v "$PWD/benchmarks/experiments/vllm-033-upgrade:/measure:ro" \
  -v "$PWD/benchmark-results/pti-check:/results" \
  --entrypoint python local/b70-qwen38-vllm:direct-xpu-development \
  /measure/profile-graph-replay.py --out /results/replay
```

The output directory must be fresh. Passing requires bit-identical eager and
graph outputs before and after profiling, changed-input replays and at least
nine real GPU kernel records from three profiled graph replays.

## Operational reference finding

The first fresh v7 reference passed API/media, near-limit context, C4/C16,
long-image, abort/extension, structured-output and 32 maximum-area image
checks. The later natural-retention test returned the same correct answer but
recorded zero CPU hits and zero H2D bytes. That is a
[failed RAM-retention measurement on the old build](qualification/baseline-mixed-load-ram-failure.json),
not evidence of a 2.15 regression. The [fresh-cache v7 repeat](qualification/baseline-isolated-ram.json) passes:
44,800 host tokens, 1,716,649,984 H2D bytes, identical correct answer and no
preemption. The original mixed-load miss remains recorded; bounded cache
retention is not guaranteed after arbitrary prior workloads. Both arms use
the same fresh-worker ordering for the final comparison, with unchanged
assertions. The candidate result remains pending.

## Fresh coding-task reference

The unchanged v7 reference solves **3/4** frozen tasks. One dependency-planning
solution raises `KeyError` instead of the requested `UnknownTarget`; its other
eight hidden checks pass. [All task results](qualification/baseline-agents/summary.json)
are retained, with no repaired code, hidden feedback or selected replacement
run. The [pre-candidate addendum](qualification/contract-addendum.json) records
that the reference4/4 target failed; the candidate still must solve all four
cases. Candidate task results remain pending. The full serving measurement
runs separately after this failure, preserving the same arm's compiler caches.

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

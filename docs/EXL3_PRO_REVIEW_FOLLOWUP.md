# EXL3 production review follow-up — 2026-10-02

This document preserves the pre-release review experiment and its identities.
The subsequent final-image qualification and deployment are in the
[v2 release report](EXL3_RELEASE_V2_REPORT.md).

The supplied Pro review identifies bounded follow-up work. Its patch is a
candidate, not an instruction to overwrite the qualified image. Work takes
place in `work/exl3-review-20261002` in separate owned and EXL3 worktrees.
No root installation or driver change has been needed.

## Release identities and scope

- Qualified production at the time of this experiment: `sha256:c09015ce22180fbc90ef0f5070f4a7116c8d11be785067499477accf0216f21f`.
- Final review candidate: `sha256:0e711fea1f9231a25289d812fffbde51ed93cbe7bad16c34f7fde3edf3d91737`.
- Candidate tag: `local/b70-qwen38-vllm:exl3-review-cache-c4copy-v2`.
- First unrestricted-copy experiment: `sha256:ed4a6deeee37c2da69829824b29fbc942d258325b03a6c87be4121cdcb923a18`.
- Owned base commit: `d4b59ae23dbad9b3b31a2f029286c86c66f85fb0`.
- EXL3 base commit: `81ced069d3953b8f7d320aadfdb50808ec44c771`.
- Committed candidate source: `c59d9442aba8610188837e37724600f1517d7335` ([pinned snapshot](../engine/exl3xpu/review-candidate/manifest.json)).
- Native attention formulas, exact K lengths, quantization, MTP3, draft
  vocabulary and existing numerical thresholds remain unchanged.

At the time of this experiment, the candidate was not a promoted release. Its native library is a
fresh build with a different manifest, and needs its own release attestation.
The v1 image, source snapshot and qualification receipts retain their identity.

## Completed checks

### F1: operational helpers

`build-image.sh` uses `B70_BASE_BUILD_IMAGE`, independently of the serving
`.env`, and rejects current/rollback release tags before downloads or Docker
mutations. `restore-production.py` checks current qualification receipts,
the user unit's working directory/ExecStart and the actual strict launcher
before starting the service. Seven focused mock tests pass, including rejection
without side effects. No GPU benchmark is needed for these fixes.

### F4: runtime source closure

`collect-review-runtime.py` derives the union of the active installed
`SOURCE_HASHES` declarations. Missing dependencies, conflicting declarations or
hash mismatches fail the capture. The actual qualified container supplied
13/13 guarded files, including the V2 runner, attention utilities and hybrid
Mamba model state. A compact supplemental ZIP contains this closure and the
full V2 worker subtree, with installed/workspace relationships and provenance.
Disposable CPU-only containers also prove rejection when the V2 runner source
is missing or has a different hash. No qualified image is modified.

### F2: application partition cache

The candidate uses a bounded LRU of exact-shape compiled oneDNN partitions.
Default capacity is 64 (`EXL3_SDPA_CACHE_CAPACITY`, integer 1–256). Hits do not
wait for GPU completion. Eviction first chooses a completed partition; only a
full cache with all entries still in flight waits for the oldest completion.
Every use retains its completion event and chains it to prior uses. The
`exl3_sdpa_cache_stats` operation exposes entries, capacity, hits/misses,
evictions, pressure waits, compile microseconds and peak entries explicitly,
outside the serving hot path. oneDNN's internal caches are a separate scope.

The host cache policy test covers in-flight eviction, busy pressure, hits and
compile failure. Actual B70 finite soaks use 81 exact KV lengths over three
epochs, with repeats/hot-shape reuse, at capacity 8 and default 64. Both obey
their caps, show hits and eviction, and pass the original numerical threshold.
From epoch two to three, RSS growth is 978,944 / 958,464 bytes respectively;
XPU allocated/reserved growth is zero. No actual pressure wait occurred in
these finite GPU runs; that branch is covered by the host event-policy test.
This demonstrates bounded retention in this workload, not an indefinite OOM
guarantee for all allocators or dependencies.

### P1: direct M04 output copy

The supplied unrestricted patch was tested, then limited to C4 in the final
candidate. It copies the permuted view directly into caller-provided output;
C1–C3 retain the qualified unpack/copy path and the no-output fallback remains.
The CPU layout test, 16 C1/C4 × q2–5 × KV4097/102753 GPU cases, original native
tolerance, bitwise old/new comparison and changed-input graph replay pass.
Output storage identity is preserved. Initial eight-round timings were noisy.
A bounded q4 repeat uses 20 ABBA rounds with 100 cycles/event. At KV4097, C4
paired mean gain is 0.762% (paired-round bootstrap 95% interval 0.678–0.846%);
C1 is 0.536% slower (interval -0.882 to -0.282%). Long-context intervals include
zero. This is why the general patch is not retained. The final C4-only source
passes all 16 numerical/changed-input graph cases again. This remains a
copy/kernel micro-test, not a measured serving throughput improvement.

### P2: single guarded route decision

The candidate determines the guarded route once for logging and execution.
Mixed subgroups still get their own checked decision. The focused attention
gate passes 31 cases, two mutable graph cases and three future-token poison
cases under the unchanged rtol 0.01 / atol 0.003.

### F3: matched histories, full target logits and real batch states

Eight bounded target-step comparisons cover actual ≥4096-token prefill, an
8192-token prompt, mixed prefill/decode, real C4 × q4 verification, and the
two requested output positions: request index 2/output index 1 and request
index 3/output index 3. The diagnostic hooks the hash-guarded V2 target call.
Inputs and metadata are restored, as are all addressed KV/GDN blocks; the
mixed and exact early-position follow-ups additionally restore the entire
11,196,825,600-byte underlying cache storage and verify its byte hash.

Native, native-repeat, M04-only, oneDNN-only and combined routes produce full
248,320-vocabulary target logits on the actual sampled batch rows. Real C4
verification is replayed through newly captured diagnostic XPU graphs with
the actual inputs/metadata. 256 local same-input attention comparisons pass
the unchanged rtol 0.01 / atol 0.003. Native-repeat is bitidentical for the
tested pure C4 graph; M04 native-relative mean KL is about 0.0000109 in one
16-row verification batch, with no top1 change. This KL compares attention
routes; it is not a new BF16 quantization-quality metric.

The early mixed native-repeat itself varies even after whole-cache restoration.
A further bounded capture locates tiny differences before attention: the
first GDN b/a projection differs by at most 0.001953125 (RMS 0.00000446), while
its qkv projection is bitidentical. Differences propagate through GDN and later
layers. Mixed segmentation also changes the native decode subgroup's call
shape. At the two specifically captured early positions, no route changes
the winning token; one winner margin is only 0.0078125 and changes across
attention routes. These observations support numerical sensitivity in the
tested pipeline; they do not reconstruct all nine historical divergent texts
or prove universal determinism. No failed local numerical gate or evidence
requiring route disablement was found. No tolerances were widened.

These shadow calls deliberately perturb timings and cannot enter throughput
tables. The original BF16 reference was reused; no new reference generation,
MTP/vocabulary campaign or complete 70-wave benchmark was needed.

### Documentation and review packaging

README now documents the current service, defaults, limits, measurements and
rollback. GPTQ comparisons, coding detail and reproduction commands live in
separate documents. Its serving section reproduces the existing validated
summary byte-for-byte; candidate microresults do not replace those numbers.
The renderer updates only the serving measurement markers, preserving the
operations guide. `build-review-bundle.py` includes the complete current owned
and EXL3 candidate text source, the qualified EXL3 source separately, all active
guarded runtime dependencies and compact diagnostic receipts. Required source
must pass hash checks. Weights, binaries, raw logits, duplicate progress records
and selected older per-position metrics are explicitly omitted.

## Release boundary and remaining promotion work

The bounded review implementation/tests are complete. Kernel split retuning
was not pursued. Host helpers and documentation require no image replacement.
The final candidate runtime has its own image identity and reuses only the
exact native ABI/source/hash-verified library (SHA256
`e60e499aefa95b290924ed98545eb815aca9d0fb1f3ede39b37d1ad49f3b522d`).
Before runtime promotion it needs a distinct release policy/attestation and
focused serving/restart qualification with those bytes. Existing v1 full
benchmark numbers are not relabeled as measurements of this candidate.

The [compact validated assessment](../benchmarks/results/exl3-review-20261002/assessment.json)
preserves campaign/image identities, source hashes, all timing arms and the
scope of the numerical checks.

## Local evidence and reproduction

Raw evidence is under `benchmark-results/pro-review-20261002/`:
`candidate-build-c4copy-v3`, `runtime-sources`, `focused-xpu-v1`,
`m04-q4-repeat-v1`, `m04-c4copy-final-v1`, `matched-state-v5`,
`matched-state-full-mixed-v1`, `matched-state-native-locate-v1`,
`matched-state-early-targets-v1`. Failed diagnostic setup attempts remain local;
they concern alias handling, a snapshot memory guard and byte hashing, not a
GPU numerical gate. Exact final early-target scripts match their launch hashes;
future runs mount immutable diagnostic source snapshots.
The first native build failure is retained (const completion-event wait); the
fresh second build passes. The initial micro-controller's service restoration
check ran too early for Type=exec; recovery verified the healthy model endpoint
and unchanged production image. The controller now waits for actual readiness.

```bash
python3 -m unittest discover -s tests/unit -p test_operational_helpers.py -v
python3 scripts/collect-review-runtime.py --output benchmark-results/review-runtime-new
python3 scripts/run-exl3-review-micro.py \
  --image local/b70-qwen38-vllm:exl3-review-cache-c4copy-v2 \
  --out benchmark-results/review-micro-new
python3 scripts/run-exl3-matched-state.py --out benchmark-results/review-state-new
python3 scripts/run-exl3-matched-state.py --early-targets --out benchmark-results/review-early-new
```

GPU diagnostics require exclusive use of the card. They restore the original
qualified service and verify its image and healthy models endpoint afterward.

# EXL3 migration execution notes

The authoritative scope is [the Pro implementation plan](EXL3_PRODUCTION_MIGRATION_AND_OPTIMIZATION_PLAN.md).
The active migration branch is `work/exl3-migration-20261001`; the EXL3 worktree is
`../exl3xpu-migration`. GPTQ remains the production release. No production tag or
launcher has been changed. Release requires all gates and explicit acceptance
of any remaining performance regression.

## Baseline — 2026-10-01

`benchmarks/results/exl3-migration/baseline_environment.json` captures source,
runtime, checkpoint and loaded native-library identities. Original benchmark
JSON is copied byte-for-byte into `baseline-evidence/`; these are historical
results, not measurements of the migration branch. The EXL3 image was inspected
without starting a GPU worker; its complete loaded-library map is still required
when the production-contract candidate runs.

The two vLLM trees in the GPTQ image differ. The `vllm` console script resolves
`/opt/venv/lib/python3.12/site-packages/vllm`; `python -c` from the image working
directory resolves `/workspace/vllm/vllm`. Workers inherit the console script's
search path. Tests and patches must identify the launch path rather than assume
that an interactive import describes deployed serving. The container command,
console search path and hashes of both code trees are recorded.

The pinned production image is
`sha256:ed1ebca756abb0e0832d11cd0db026dd7e86df094c6903efe7ae8afbdc290b68`.
The old EXL3 image is
`sha256:cba73584f4ab0a2b37eac1356f34f16655ac5e740d845e110997b03be78279b7`.
Power was verified at 180 W. Runtime versions remain Torch 2.13/vLLM 0.30 for
GPTQ and Torch 2.12/vLLM 0.26.1 for the initial EXL3 baseline.

## Safe foundation

EXL3 compilation now uses a private temporary library, full compiler logs,
`pipefail`, registration/capability smoke tests and atomic installation. The
Torch C++ ABI is obtained from the build environment. The adjacent manifest
records compiler, flags, sources, Torch, driver/IGC, oneDNN and library hashes.
An actual invalid `icpx` flag failed with exit 1 and preserved both the previous
library and its manifest byte-for-byte. The successful Torch-2.12 build exposes
all 18 expected operators. This library must not be reused for Torch 2.13.

The loader and bit-exact test share an authoritative `.trellis` inventory. The
checkpoint contains 409 quantized modules: 408 at 4 bpw, one target LM head at
6 bpw, including eight MTP modules absent from `tensor_storage`. Missing
components, contradictory formats/codebooks and duplicate canonical names fail
early. EXL3 configuration rejects all three GPTQ draft overrides. The separate
GPTQ runtime patch selects overrides only for GPTQ formats and rejects an INT4
MTP conversion without unquantized source weights. The existing production
combination (BF16 construction followed by INT4 draft conversion) is supported.

Native results in `benchmarks/results/exl3-migration/safe-foundation/`:

| Check | Result |
|---|---|
| Bit-exact reconstruction and one-hot GEMM | PASS, 409/409 modules; 8/8 MTP; no skips |
| INT8 prefill, M=8192, independent Torch INT8 reference | PASS; maximum relative error 0.0002695601, unchanged tolerance 0.002 |
| Deliberately perturbed INT8 reference | Expected FAIL/exit 1; maximum relative error 0.0909117 |
| Deliberate actual compiler error | Expected exit 1; installed library and manifest unchanged |
| GPTQ restoration after exclusive GPU tests | PASS; original pinned image, HTTP health 200 |

These tests prove the tested native reconstruction/linear paths. They do not
qualify the new loader on a complete serving model, attention, the 0.30 port,
FP8 cache quality, graphs, long-context capacity or throughput.

Next: complete loader reports, qualify the existing EXL3 image under the
200,704-token/C4/media/API contract, and instrument scheduler/cache allocations
before diagnosing preemptions. The runtime port and performance experiments
follow those measurements, in the order specified by the plan.

## Old-runtime contract and preemption diagnosis

The initial safe old-ABI candidate is a separate image,
`sha256:f07d223d6de11e72f7ac00006c3697cd3662267d8938f9c5c0c84a476797cad9`.
The campaign's Docker inspection records the full image identity. The new profile
uses 200,704 total tokens, C4, 6,656 batch tokens, memory fraction 0.93, FP8 KV,
prefix alignment, one image/no video, the production API alias and parsers.
MTP remains at the old EXL3 depth 3 and its 65,536-row draft vocabulary. These
instrumented capacity runs are not throughput qualification.

The model loader audit passes 409/409 modules, including 8/8 MTP modules and
the 6-bpw target/draft head. CPU integration tests validate QKV's independent
Hadamard transforms, GDN's shared QKV transform and separate z transform,
head slicing, missing/duplicate/invalid shards and TP rejection. Auditing is
installed on the worker's load method so that both V1 and V2 runners are covered.

`old-contract-smoke-v1/` passes text, streaming, automatic function calls,
reasoning, a red-image prompt, rejection of two images, prefix extension,
client disconnect/restart, and four simultaneous small requests. Prefix
extension reuses 3,200 tokens. This is not a long-context C4 capacity test.

`old-contract-long-v1/` uses a fresh worker for every case:

| Prompt / output tokens | Traced preemptions | Scheduled prefill tokens |
|---|---:|---:|
| 102,752 / 1,024 | 0 | 102,752 |
| 139,193 / 1,024 | 0 | 139,193 |
| 187,695 / 1,024 | 1 | 194,095 |
| 199,680 / 1,024 (total boundary 200,704) | 1 | 206,080 |

The failures are cache-pool exhaustion, not observed device OOM. At the
188K rejection, 7 blocks are requested (1 per each of 3 GDN groups plus 4 full
attention blocks), but only 4 are free. The 176-block pool holds 26 non-null
states per GDN group and 93 full-attention blocks, plus its null block. Driver
free memory remains several GB; increasing the memory fraction would hide
the faulty retirement rather than correct it.

The pinned old allocator's `_remove_blocks_in_range` walks backward and
**breaks at the first null entry**. GDN block tables have sparse holes with
older live states behind them. Those states retain request references until
completion/preemption and consume one extra state per prefill chunk. The
139K case fits after the larger production chunk budget; the 188K and boundary
cases still exceed the pool. This explains why contract differences matter.

After the 188K preemption, the cache lookup resumes at 140,800 tokens. The
scheduled-work trace adds one 6,400-token chunk, not a full 187,695-token replay.
The logical prompt/prefill metrics still report 187,695 and initial cached tokens
0. Scheduled work is not a completed-GPU-compute measure. An additional worker
forward trace is being collected to distinguish actual submitted work from
optimistic scheduling and cancelled work.

The proposed fix changes only the old `MambaManager`: continue across null
holes inside the **existing processed-token retirement boundary**. Current,
speculative and in-flight state protection and reference counts stay intact.
It is guarded by the exact allocator source hash and an opt-in candidate flag.
Five CPU tests pass, including the demonstrated old sparse-hole failure,
protected boundaries, shared references, repeat retirement and Mamba-only scope.
The fixed GPU campaign and matched greedy-output check must pass before the
runtime port begins. No performance optimization has been promoted.

## GPU availability policy — user update 2026-10-01

The user explicitly requested that GPTQ remain offline during EXL3 development.
Campaigns now leave the service stopped by default; `--restore-production` is
reserved for an explicit rollback drill (web campaign: `--restore-after-campaign`).
GPTQ is loaded only for selected comparison tests or the final rollback gate.
The production image, model revision and launcher remain pinned and unchanged.
The already-running operational campaign loaded its previous cleanup code before
this instruction; a completion watcher stops GPTQ immediately after that cleanup.

## Old-runtime sparse retirement — GPU confirmation

`old-retirement-control-v2/` and `old-retirement-fixed-v2/` use the same immutable
image `sha256:3fcacd9c4971356b228fe638151b1d804e22fa5e7438f57a79864948068aa33f`,
changing only the retirement flag. The 22,450-token matched greedy request and
response message SHA256 match exactly. Fixed API smoke passes.

| Prompt / output | Preemptions with fix | Worker-submitted prefill tokens |
|---|---:|---:|
| 102,752 / 1,024 | 0 | 102,752 |
| 139,193 / 1,024 | 0 | 139,193 |
| 187,695 / 1,024 | 0 | 187,695 |
| 199,680 / 1,024 | 0 | 199,680 |

Actual Worker.execute_model tracing confirms that the unfixed 188K case submits
194,095 prefill token positions: one extra 6,400-token chunk. The fixed case
submits exactly the logical prompt length. GPU completion remains asynchronous;
these observations do not replace component-level performance timing.
Long C4, long image and long prefix/abort operational cases are still being
qualified before the target-runtime GPU port begins.

Old-runtime operational qualification is now PASS: four simultaneous independent
32,768-token prompts with 256 output tokens each, one image plus 119,093 prompt
tokens, and a 50K prefix extension reusing 48,000 tokens all have zero preemptions.
The aborted request is explicitly traced as FINISHED_ABORTED and a subsequent
request succeeds. GPTQ is now stopped; no model is resident between campaigns.
The target installed 0.30 code already rounds GDN pages to multiples of 64 and
scans sparse Mamba holes (with a retired-prefix watermark). Old platform and
allocator overrides must not be installed there.

## Serving-profile preference — user update 2026-10-01

The user prefers the useful upstream 0xsero EXL3 serving settings rather than
permanently retaining GPTQ limits. The Pro plan section 7 mandates 0.93 /
200704 / C4 / one image as an initial comparison contract, not a final
EXL3 ceiling. Its instruction not to raise the memory fraction merely to
force a pass still explains the unchanged diagnostic campaign. The user
steering establishes the expanded profile as the intended next qualification.

| Setting | Port comparison profile | Upstream / desired candidate |
|---|---:|---:|
| GPU memory fraction | 0.93 | 0.965 |
| Total context per request | 200704 | 262144 |
| Max active sequences | 4 | 16 |
| Prefill token budget | 6656 | 4096, pending mixed-serving comparison |
| Media count per request | 1 image / 0 video | 32 images / 4 videos |
| Image preprocessing area cap | 4194304 pixels | 4194304 pixels |

Candidate: EXL3 worktree `models/qwen3.8-27b-exl3-4.00bpw/
migration-target-upstream-expanded.yaml`. It retains the new-runtime MTP
method, source-hash/loader guards, API alias/tool/reasoning/defaults and
modern pixel-cap spelling. It is prepared, not GPU-qualified or deployed;
no existing equal-contract result proves expanded capacity. The active
image predates this additional YAML file.

Finish the running equal-contract port campaign. Then isolate .965 at the
existing C4/context/media settings, qualify expanded media using actual
image/video requests and multimodal memory profiling, and qualify 262144
context plus C16 admission. C16 is not a promise that sixteen independent
262K contexts fit concurrently. Raising media limits can change startup
vision-memory reservations; the extra approximately 1 GB of budget is not
a guaranteed KV gain. Keep the image-area cap to bound encoder work.

After staged numerical quality and the plan's first MTP3/MTP4 performance
experiment, compare the upstream 4096 prefill budget with 6656 using matched
mixed-serving requests and decode latency percentiles. Smaller chunks may
reduce decode wait while changing prefill throughput. Do not silently
choose a winner from old-runtime measurements.

Keep beneficial EXL3 features (FP8 KV, prefix cache, graph capture, pruned
MTP head and validated INT8 prefill) subject to their separate gates. Port
oneDNN/verify attention separately. Do not copy old alignment/allocator
patches onto 0.30, or enable the upstream two-GPU data-parallel example on
this single-B70 setup. GPTQ stays offline.

## Target-runtime port — current evidence

Fresh Torch 2.13 native compilation passes all 409 reconstruction/linear
checks, the INT8 numerical gate and deliberately injected test failure. An
actual compiler failure preserves the installed library and manifest.
Seven-site GDN semantic conversion passes CPU/XPU and graph checks. The
active worker is XPUModelRunnerV2; full loader audit covers all 409 modules
and all eight MTP modules.

An initial warm-cache startup failed at the unchanged .93/200704 budget.
Startup allocation tracing found discarded full/pruned EXL3 LM heads held
by loader callback closures. Weak references remove those owners without
changing model tensors, freeing 1209132544 bytes. Both cold and warm
startups now pass at the original comparison budget; matched greedy
output and API smoke pass. The current candidate is
`sha256:046ef4c7937d0e6f8945191671d7147270f7e24f1772159c48b395ad47451088`.
Long/operational qualification remains running in
`target-contract-weakref-v4`; this is not throughput or full quality
qualification.

Target functional qualification has now completed. All nine cases in
`target-contract-weakref-v4` pass with zero traced preemptions and allocation
rejections; the long abort is explicitly FINISHED_ABORTED. The frozen greedy
request and output hashes match the fixed old runtime. See
`target-port-v1/target-contract-proof.json` and the per-case trace summaries.
The port commit is EXL3 `69797ba`; no performance optimization is promoted.

The short-panel quality campaign is now running at
`benchmark-results/exl3-quality-target-stages-v1` under exclusive GPU lock.
Five separately identified arms progressively add FP8 KV, INT8 prefill,
graph configuration and MTP3 to the new FP16-KV port. They use identical
1024-token windows and native full-vocabulary capture with independent API
NLL alignment, FP64 normalization and paired window bootstrap. The staged
short-panel scoring alone does not qualify generated decode graphs,
speculative acceptance, long-context quality or the expanded production
profile. Those remain explicit gates.

Baseline vocabulary audit finds 248077 contiguous tokenizer IDs versus
248320 model logit rows. No additional masking is applied to any arm.
Mean probability mass on the 243 other rows is approximately 3.25e-8 for
BF16, 4.35e-8 for GPTQ and 3.58e-8 for old EXL3; recomputed baseline PPL/KL
match preserved measurements. See `baseline-vocabulary-audit.json`.

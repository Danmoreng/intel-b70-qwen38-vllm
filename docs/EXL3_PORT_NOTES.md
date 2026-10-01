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

## Staged short-panel quality — completed

The corrected campaign `exl3-quality-target-stages-v2` reuses the first
four completed arms and runs only the missing MTP arm. The initial MTP
startup failed because the measurement runner pointed at a nonexistent
checkpoint sidecar. The draft vocabulary actually resides in the image's
model-profile directory; this runner error and failed campaign are preserved.
All five arms now pass native capture/API NLL alignment for all 16368
positions and provide 512 full-vocabulary distributions. Original arrays
are bit-exact to the frozen BF16 bundle, including IDs and sample positions.

| Arm | Mixed-panel PPL | KL(BF16 || arm), nats |
|---|---:|---:|
| Old EXL3 FP16 reference arm | 3.641980 | 0.031493 |
| New port, FP16 KV / eager | 3.641900 | 0.031457 |
| Add FP8 KV | 3.642457 | 0.031522 |
| Add INT8 prefill | 3.645656 | 0.033853 |
| Add graph configuration | 3.646181 | 0.033254 |
| Add MTP3 configuration | 3.647060 | 0.033893 |
| Existing GPTQ | 3.801927 | 0.086369 |

Values are recomputed from stored arrays with FP64 normalization. The
MTP-configured arm has PPL +1.180% versus BF16, compared with +5.476%
for existing GPTQ. These results retain the short-panel EXL3 advantage;
The MTP-configured PPL is 0.1395% above old EXL3. Its paired NLL
delta is 0.001394 nats/token, exploratory 95% window-bootstrap interval
[0.00000891, 0.002804]; this small effect is retained in the report rather
than treated as zero. No new arbitrary release threshold is introduced.
Short teacher-forced
scoring does not itself qualify generated MTP acceptance/graph execution
or long-context precision. Gate E remains open for those checks and the
full expanded candidate profile. Lean acquisition/provenance/metric JSON
is committed under `target-quality-stages-v1`; full arrays remain local
and have a SHA256 manifest.

The requested .965 profile qualification now runs independently using the
immutable functional-port image plus tracked JSON overrides. The first
campaign changes only GPU memory fraction and repeats the frozen greedy
probe plus the 200704-total boundary. Media/context/concurrency changes
follow as separate cases; no unchanged full GPTQ benchmark is running.

## User steering: concurrency pressure and full draft vocabulary

The user accepts preemption under genuine capacity pressure at C16 and allows
C4/C8 as the qualified concurrency target. C16 is an admission ceiling, not
a promise of sixteen simultaneous maximum-length contexts. Retain completion,
correctness, accounting, no-OOM and recovery gates; report preemption counts,
recompute work and latency. Investigate avoidable pressure without making
zero-preemption C16 a mandatory release gate. The first expanded campaign
failed its former zero-preemption assertion after all sixteen 8192+256 requests
completed (four preemption events). Its remaining image/abort cases did not run;
this failed campaign is preserved and cannot be relabeled as complete. Future
controllers support explicit `--allow-c16-preemptions` with recorded policy.

M04 shared-KV verification is not yet active in the target EXL3 runtime.
It reuses historical KV loads across multiple speculative verification rows,
with correct causal visibility per row. Porting requires actual EXL3 page
size/strides/scales and graph/native fallback validation, not the GPTQ 1664
page-size assumption.

Current EXL3 draft pruning keeps 512 blocks of 128 tokens: 65536 draft rows,
not 512 tokens. Target verification retains the full 248320-row model head.
GPTQ production uses full draft vocabulary. After the plan's first controlled
MTP3/MTP4 experiment, compare EXL3 pruned versus full draft vocabulary with
the same selected MTP depth, prompts, output budgets, KV precision and runtime.
Disable `EXL3_DRAFT_VOCAB` for the full arm and verify actual full-head dispatch.
Measure end-to-end decode, accepted tokens per round, draft-head time, startup
and steady-state memory, and retained context capacity. Fewer draft rows can
reduce head cost but reduce acceptance; choose from measured net benefit.
Disabling pruning removes the additional pruned-head allocation but increases
full-head computation; do not assume it costs additional weight memory.

## Expanded capacity — completed under the clarified pressure policy

`target-memory965-v1` completes the unchanged C4/200704 contract at .965
with no traced preemptions. `target-media965-v1` verifies actual 32-image
and 4-video count limits, over-limit rejection and the 4.2MP image cap
(an oversized image produces4081 prompt tokens). This does not guarantee
32 maximum-area images plus4 long videos and maximum text all fit.

`target-upstream-expanded-v2` reuses the identical image/settings' completed
regression, smoke and261120+1024 boundary cases with checked runtime
manifest hashes. Fresh C16, image119K and long extension/abort/restart
cases pass. One FINISHED_ABORTED event is recorded. C16 completes all16
8192+256 requests, with4 preemption events affecting2 distinct requests.
Target-forward submission offsets account for147072 prefill tokens versus
131072 logical prompt tokens:16000 additional submitted rows. This is
submitted work, not completed GPU kernel timing. Waiting admission
rejections are not themselves request preemptions. All other cases have
zero traced preemptions. Full candidate precision and performance gates
remain open; native diagnostic timings are not release throughput claims.

Terminal trace files, including the earlier failed campaign, are compressed
losslessly with per-file roundtrip/SHA256 receipts. Reused directories use
relative links to preserved evidence in the same repository.

The first chunked original BF16 GPU control finishes in75.27 seconds using
2.57GiB peak allocated GPU memory and5.75GiB peak host RSS. With256-token
chunks, mean full-vocabulary KL from the existing unchunked BF16 reference
is0.0006482 for prose and0.0006462 for code; top1 agrees at all17 sampled
positions. Individual suffix target NLL changes reach0.225nats. This is
numerical reference drift, not quantization error or proof of long-context
accuracy. CPU official cached-layer checks pass at<4.2e-7 logit difference.
A4096-token control and actual untruncated long references follow.

## Expanded full-candidate short precision — completed

`exl3-quality-target-stages-v3` reuses five immutable completed arms and
adds only the expanded full candidate at262144/C16/.965/4096/media32/4,
FP8 KV, INT8 prefill, graph configuration and MTP3. All16 windows and
16368 target NLL positions align with the independent API;512 full
vocabulary distributions are compared with FP64 renormalization.
The frozen BF16 bundle still validates bit-exactly.

Full candidate PPL is3.647854, KL(BF16||candidate)0.032727nats; GPTQ
is3.801927/0.086369. The candidate retains the measured short-panel
EXL3 advantage. Paired deltaNLL versus oldEXL3 and its exploratory
window-bootstrap interval are preserved in `target-quality-full-v1`;
no effect is treated as zero or assigned a new post-hoc pass threshold.
Configuring MTP/graphs during prompt scoring still does not prove
generated acceptance/graph correctness. Long suffix and generated
decode checks remain before Gate E can close.

The4096-token control also finishes (73.24 seconds): on these1024-token
windows it uses one cached prefill per layer and reproduces all512 suffix
NLL values and17 sampled full-vocabulary distributions exactly (KL0).
The256-token control therefore isolates a chunking numerical effect, not
a layer-streaming/alignment defect. Long contexts necessarily use multiple
chunks; their reference metadata records that limitation. The four original
BF16 contexts completed without truncation in `exl3-long-quality-v1`:
32K/100K/180K/262K prefixes plus128 fixed suffix tokens, in2020.82 seconds,
with8.10GiB peak allocated GPU memory and5.93GiB peak host RSS. The original
short BF16 corpus was reused; these are additional long-context regression
probes, not a replacement corpus or an adaptive coding benchmark.

C16 pressure is explained by the shared hybrid block pool, not by sixteen
maximum-length contexts: the recorded admission round requests4 blocks per
GDN group (three groups, including3 speculative states each) plus6 full
attention blocks for an8K prompt,18 pooled blocks in total. Only200 of201
blocks are initially free. At a running prefill boundary, a request needs
one additional block in each group (4 total) while the pool has0 or1 free;
the scheduler preempts to make progress. This establishes block-pool
pressure and hybrid/speculative overhead. It does not prove that every
preemption is unavoidable or that improved admission could not reduce it;
no speculative allocator rewrite is required before quality/performance
work under the clarified user policy.

## Long prompt measurement repair and first MTP protocol

The initial native long-quality acquisition stopped after3200 prompt rows.
Two scoped EngineCore stack samples located a blocking extra GPU-to-CPU
request-ID read in the capture hook. Replacing that read with the original
CPU request registration and the batch CPU prefill offset exposed a device
OOM on the next prefill step. The measurement-only prompt scorer materializes
1024-row full-vocabulary logits and normalization/top-k temporaries, unlike
normal serving; the .965 KV reservation leaves limited space for that work.

The capture helper now clones the scorer with128-row head chunks and leaves
its original module helper unchanged. Scoring target IDs and normalization
remain intact; immutable engine image, weights, full context and all profile
settings are unchanged. CPU tests cover split prefill, head boundaries and
the final unscored token, forbid extra device request-metadata reads, and
reject changed frozen inputs. Acquisition `exl3-long-quality-v3` completes
all four contexts in1075.48 seconds, with27.58GiB peak allocated and27.94GiB
peak reserved GPU memory. Every native prompt NLL position is covered exactly
once and aligns with the independent API. The512 suffix target positions
give pooled PPL1.209323 versus original BF161.210748, mean full-vocabulary
KL0.0015101 over32 positions, with top1 agreement at all32. Per-window values
are preserved in `target-long-quality-full-v1`. These engineered repeated
prefixes are precision regression detectors; slightly lower pooled PPL does
not rank general model quality, and the streamed-reference numerical caveat
remains. Generated decode/graph checks remain before Gate E can close.
Both failed runs and the stack/source evidence are preserved. Frozen BF16
arrays were copied with SHA checks into retries; they were not recomputed.

The first MTP study is prepared in `target-mtp-protocol-v1`, with eight
frozen code/prose inputs at4K/32K/103K/128K, C1/C4 and cold/warm cache modes.
Only depth3/4 changes in an ABBA order; sampling and512 output-token budgets
are fixed. The103K code input is the existing actual coding history rendered
with tools and preserved reasoning. Three local HTTP/SSE integration tests
verify accounting and reject unsupported/mismatched token-ID returns. No GPU
performance samples have been acquired yet. Component/graph-row profiling
and the requested full/pruned draft-vocabulary comparison remain separate
requirements; the4K pilot cannot select a universal depth.

That pilot now completes20 measured requests, with no preemptions and all
returned prompt-ID hashes independently verified in the saved SSE logs.
MTP4 improves C1 in both cache modes; C4 results differ by cache mode. This
is one pass per point, not a robust depth selection. Generated sequences
vary across depth/batching/cache (first divergence at137 or later in the
512-token4K code output); the hashes/first differences are retained and
generated numerical/quality checks must explain that variation rather
than assuming bit-exactness. Both depths use full target vocabulary and
the same65536-row draft head.

The serving ABBA study uses `target-mtp-protocol-v2`, preserving all eight
original window dictionaries and adding48K code/prose. C4x103K/128K may
exceed the shared KV pool; pressure results remain in the study, explicitly
classified instead of treated as clean kernel throughput. The48K C4 point
is a candidate for a clean long comparison; fit is not presumed. Four HTTP
tests now also reject an API that omits prompt-ID verification. Full study
results and actual concurrent/padded rows remain pending.

A guarded diagnostic event profiler is prepared against the exact V2 source
hash, with actual call-site counts and four CPU event-plumbing branches
validated. It times target body/head, verification sampling, sampled-token
commit, complete draft proposal and full runner-cycle timeline spans, with
CPU actual/padded rows and graph bucket metadata. Event synchronization is
deferred until generation finishes. It is not an uninstrumented throughput
arm; fused draft body/head/sampler decomposition remains explicitly missing.
Actual XPU-event validation and profiling follow the completed serving study.

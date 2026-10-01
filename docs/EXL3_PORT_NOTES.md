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

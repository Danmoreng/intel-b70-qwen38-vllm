# B70 / Qwen3.8-27B EXL3 Production Migration and Optimization Plan

**Date:** 2026-10-01  
**Target:** Intel Arc Pro B70, Qwen3.8-27B, vLLM-based serving  
**Current production baseline:** GPTQ INT4 / G128  
**Candidate:** EXL3 4 bpw with 6 bpw target LM head  
**Primary goals:** preserve or improve model quality, retain the existing production serving contract, exploit EXL3's lower weight footprint, and close as much of the decode-throughput gap as possible without compromising numerical correctness.

---

## 1. Executive decision and scope

EXL3 should be treated as the **preferred quality candidate**, but it should **not replace the current GPTQ production engine yet**.

The existing evidence shows a substantial quality advantage for the supplied EXL3 checkpoint over the supplied production GPTQ checkpoint, while the clean long-context serving comparison still shows a meaningful decode-throughput deficit and the longer EXL3 runs exhibit preemptions that are not yet explained.

The migration should therefore proceed as a gated engineering program:

1. make the EXL3 build and loader safe for production work;
2. align EXL3 with the current production API and capacity contract;
3. diagnose and eliminate unexplained long-context preemptions;
4. port EXL3 cleanly to the target vLLM/Torch runtime;
5. revalidate quality under the actual production precision path;
6. optimize decode in order of expected impact, starting with speculative decoding behavior rather than immediately rewriting the EXL3 Trellis kernels;
7. release only after explicit quality, correctness, capacity, latency, and rollback gates pass.

Do **not** interpret this plan as approval to switch production traffic merely because EXL3 has lower perplexity/KL or smaller weight files.

---

## 2. Evidence already established by the review

Codex should preserve these findings as the initial baseline. Do not relabel them as new measurements unless they are actually rerun on the B70.

### 2.1 Review scope already completed

The supplied repository/archive was statically and structurally reviewed across the relevant EXL3, GPTQ, runtime, attention, MTP, build, launcher, and benchmark paths.

Already completed during the review:

- SHA-256 and size verification of **1,456 manifest entries** with no mismatches.
- Python AST syntax validation of:
  - 40 EXL3 Python files,
  - 207 GPTQ Python files,
  - 233 deployed-runtime Python files,
  with no syntax failures.
- CPU synthetic capture validation for both capture runner versions, including:
  - split-prefill alignment,
  - correct next-token shift,
  - exclusion of the final unscoreable logit,
  - full vocabulary size of 248,320.
- Arithmetic revalidation of the supplied quality and performance measurements.
- Static application of the older EXL3 GDN patch logic against the supplied installed vLLM 0.30 source, revealing partial rather than complete coverage.

Not executed during the review:

- native SYCL/C++ compilation,
- B70/XPU benchmarks,
- 27B inference,
- full PPL/KL recomputation from raw logits that were not included,
- deployment of a new runtime image.

Any repository report produced after implementation must retain this distinction.

### 2.2 Quality baseline

Source evidence: `evidence/quantization-reference/comparison-foem.json`.

| Arm | Perplexity | PPL increase vs. original | Mean KL(BF16 || arm), nats |
|---|---:|---:|---:|
| Original BF16 | 3.60453 | reference | 0 |
| Unquantized FP16 | 3.60380 | -0.020% | 0.000458 |
| Production GPTQ G128 | 3.80193 | +5.476% | 0.086369 |
| GPTQ FOEM G128 | 3.80034 | +5.432% | 0.071485 |
| **EXL3 4 bpw / target head 6 bpw** | **3.64198** | **+1.039%** | **0.031493** |

On this panel, the supplied EXL3 checkpoint has approximately **63.5% lower mean KL divergence** than the production GPTQ checkpoint.

The paired GPTQ-minus-EXL3 NLL difference is approximately `0.042980 nats/token`, with the supplied window-bootstrap 95% interval approximately `[0.022841, 0.065060]`.

The code-domain PPL increase reported in the evidence is also materially smaller for EXL3 than GPTQ.

Interpretation to preserve:

- this is strong evidence that **this EXL3 checkpoint/runtime combination is closer to the BF16 reference than this GPTQ checkpoint/runtime combination**;
- it is not proof that every EXL3 quantization is universally better than every GPTQ quantization;
- it does not yet prove that the full production path with FP8 KV, INT8 prefill, graphs, MTP, long contexts, and mixed serving preserves the same advantage.

### 2.3 Clean serving-performance baseline

Source evidence: `gptq-engine/benchmarks/runs/2026-10-01-flappybird/cold-comparison.json`.

| Input tokens | GPTQ decode | EXL3 decode | GPTQ preemptions | EXL3 preemptions |
|---:|---:|---:|---:|---:|
| 102,752 | 64.45 tok/s | 46.09 tok/s | 0 | 0 |
| 139,193 | 37.52 tok/s | 33.70 tok/s | 0 | 1 |
| 187,695 | 29.81 tok/s | 24.45 tok/s | 0 | 1 |

The 102,752-token pair is the cleanest existing comparison because neither side preempted. EXL3 is about **28.5% lower in decode throughput** in that serving configuration. Relative to its own measured rate, EXL3 would need approximately **39.8% more throughput** to match GPTQ in that sample.

Do not attribute the full gap directly to the EXL3 GEMM/Trellis kernel. The speculative-decoding configurations differ.

### 2.4 Speculative-decoding baseline at 102,752 input tokens

Derived from the supplied counters:

| Metric | GPTQ MTP4 | EXL3 MTP3 |
|---|---:|---:|
| Draft tokens | 1,024 | 978 |
| Accepted draft tokens | 768 | 698 |
| Inferred speculative rounds | 256 | 326 |
| Generated tokens | 1,024 | 1,024 |
| Generated tokens / inferred round | 4.000 | 3.141 |
| Mean decode time / inferred round | 62.00 ms | 68.09 ms |

At this point, the measured serving gap is consistent with **both**:

- fewer generated tokens per speculative round for EXL3, and
- somewhat longer rounds.

This is why the first performance experiment must be a controlled **EXL3 MTP3 vs. EXL3 MTP4** study, not a speculative kernel rewrite.

### 2.5 Weight footprint and capacity baseline

The supplied weight files are approximately:

- GPTQ: **19.56 GB**
- EXL3: **16.88 GB**

The smaller EXL3 file footprint is useful, but it is **not yet a proof of additional usable context capacity** because runtime versions, graph pools, MTP representation, scheduler configuration, media limits, cache allocations, and workspaces differ.

For the reviewed architecture, pure FP8 attention KV storage is approximately:

`16 attention layers * 2 (K/V) * 4 KV heads * 256 head dimension * 1 byte = 32 KiB/token`

At 200,704 tokens, that is approximately **6.125 GiB of attention KV payload alone**, excluding GDN state, MTP state, padding/block slack, scales, metadata, graph memory, weights, and workspaces.

---

## 3. Non-negotiable engineering rules

Codex should apply these rules throughout the work.

### 3.1 Change one major variable at a time

Keep these stages separate:

1. existing EXL3 image under the current production contract;
2. capacity/preemption diagnosis;
3. clean runtime port;
4. correctness/quality validation;
5. performance optimization.

Do not combine a runtime upgrade, scheduler redesign, new MTP depth, attention kernel replacement, and EXL3 kernel tuning in one untraceable patch.

### 3.2 Preserve a pinned GPTQ rollback path

Do not modify or overwrite the known-good production GPTQ image/checkpoint in place.

Maintain:

- pinned GPTQ image identifier;
- pinned GPTQ model/checkpoint identifier;
- exact production launch configuration;
- separate graph/oneDNN/Triton/native caches when ABI/compiler/runtime versions differ.

### 3.3 No silent fallbacks or false-positive tests

A test that prints `FAIL` but exits zero is not a release gate.

A native build that fails but leaves an older `.so` behind is not a successful build.

A patch that reports six substitutions while leaving semantically equivalent synchronization sites unpatched is not complete merely because its guard passes.

Every gate must be machine-readable and fail closed.

### 3.4 Numerical correctness outranks speed

Never retain a performance change that:

- changes EXL3 weight reconstruction unexpectedly;
- corrupts causal attention semantics;
- reads/writes padded rows as valid tokens;
- introduces NaNs/Inf;
- changes the target-head 6 bpw contract without an explicit new quality study;
- changes probability masking/vocabulary semantics;
- relies on a historically known incorrect mode such as unchecked `EXL3_K4_HALVES`.

### 3.5 Do not report inferred performance as instrumented performance

Counter-derived speculative-round times are useful diagnostics, but they are not identical-state microbenchmarks.

Repository reports must distinguish:

- directly measured values;
- derived values;
- static code-review findings;
- hypotheses awaiting B70 validation.

---

## 4. Phase 0 — Establish a reproducible baseline

### Goal

Create a reproducible baseline before changing code.

### Tasks

- [ ] Record exact source revisions for:
  - GPTQ engine,
  - EXL3 engine,
  - deployed vLLM/runtime,
  - model/checkpoint artifacts.
- [ ] Record the current B70 stack:
  - GPU/driver version,
  - oneAPI/compiler version,
  - IGC version,
  - oneDNN version,
  - PyTorch version,
  - vLLM version,
  - Python version,
  - relevant environment variables.
- [ ] Record power/performance mode, including the intended 180 W profile if that is the production target.
- [ ] Record exact launch arguments for GPTQ production and current EXL3 candidate.
- [ ] Save hashes of all native shared libraries loaded by each engine.
- [ ] Save the existing benchmark JSON files unchanged as immutable baseline evidence.
- [ ] Add a small environment manifest generator if one does not already exist.

### Required output

Create a machine-readable baseline manifest, for example:

`benchmarks/results/exl3-migration/baseline_environment.json`

It should include source hashes, image identifiers, runtime versions, launch parameters, native library hashes, model artifact hashes, and B70 hardware/driver information.

### Pass criteria

- A second engineer/Codex run can reconstruct which source, model, runtime, and native libraries produced every benchmark result.

---

## 5. Phase 1 — Fix release-blocking build and validation issues

Do this before performance tuning.

### 5.1 Make native compilation fail closed

Relevant source:

`exl3-engine/scripts/build_ext.sh`

#### Problem

The reviewed build pipeline contains a pattern equivalent to:

```sh
icpx ... | grep -E "error" || true
```

This can hide compiler failure and allow a stale `_C.so` to survive, creating a false-success build.

#### Implementation

- [ ] Build to a temporary output path.
- [ ] Preserve the complete compiler output via `tee` or equivalent.
- [ ] Preserve and check the actual compiler exit status.
- [ ] Do not replace the existing working `.so` unless compilation succeeds.
- [ ] Atomically install the new library after successful compilation.
- [ ] Immediately load the library in the target Python/Torch environment.
- [ ] Verify all expected registered operators are present.
- [ ] Write a build manifest containing:
  - source hash,
  - compiler version,
  - compiler flags,
  - PyTorch version and C++ ABI information,
  - oneDNN version,
  - driver/IGC information if available,
  - resulting shared-library hash.

#### Pass criteria

- An intentionally injected compile error produces a non-zero exit and cannot leave a newly reported successful build.
- A successful build can be imported and its expected ops enumerated.

---

### 5.2 Prevent GPTQ-specific MTP overrides from touching EXL3

Relevant installed runtime area:

`vllm/model_executor/models/qwen3_5_mtp.py`

#### Problem

The current GPTQ-oriented controls can disable/replace quantization behavior in a way that is unsafe for EXL3, including behavior associated with:

- `B70_MTP_BF16_DRAFT`
- `B70_DRAFT_LMHEAD_INT4`
- `B70_DRAFT_MTP_INT4`

#### Implementation

- [ ] Make every GPTQ-specific MTP override explicitly conditional on the GPTQ quantization format.
- [ ] Ensure EXL3 never receives GPTQ-specific INT4 representation assumptions.
- [ ] For the initial EXL3 production candidate, remove/disable those GPTQ-specific environment overrides.
- [ ] Add startup logging that states which MTP quantization path is active.
- [ ] Fail startup if mutually incompatible override combinations are requested.

#### Pass criteria

- EXL3 startup logs show the EXL3 path, not GPTQ draft overrides.
- GPTQ behavior remains unchanged under the existing production environment.
- Unit tests cover both formats.

---

### 5.3 Align EXL3 bit-exact test enumeration with the actual loader

Relevant files:

- `exl3-engine/tests/test_bitexact_xpu.py`
- `exl3-engine/exl3xpu/vllm_plugin.py`

#### Problem

The reviewed bit-exact test enumerates tensors through `quantization_config.json.tensor_storage`, while the loader also discovers `.trellis` entries via the weight index. MTP tensors can therefore be loaded without being fully covered by the test.

#### Implementation

- [ ] Make the test enumerate quantized tensors from the same authoritative weight index used by the loader.
- [ ] Report coverage explicitly:
  - expected tensor count,
  - tested tensor count,
  - skipped tensor count and reasons,
  - MTP tensor count,
  - 4 bpw and 6 bpw tensor counts.
- [ ] Fail if an expected quantized tensor is not tested unless it is on an explicit allowlist.

#### Pass criteria

- 100% of expected EXL3 quantized tensors are covered or explicitly justified.
- MTP tensors are demonstrably included.

---

### 5.4 Make `test_int8_prefill.py` a real CI gate

Relevant file:

`exl3-engine/tests/test_int8_prefill.py`

#### Implementation

- [ ] Replace print-only PASS/FAIL behavior with assertions or a non-zero process exit on failure.
- [ ] Save numerical diagnostics when the test fails.
- [ ] Preserve the existing error threshold unless a separately justified validation changes it.

#### Pass criteria

- Deliberately violating the tolerance fails CI/process exit status.

---

## 6. Phase 2 — Preserve EXL3 loader semantics during all ports

Relevant source:

`exl3-engine/exl3xpu/vllm_plugin.py`

### Required invariants

Codex must preserve and test all of the following:

- `.trellis`-based module/tensor discovery from the weight index;
- 4 bpw main linear weights;
- 4 bpw MTP linear weights where supplied;
- 6 bpw target LM head;
- draft vocabulary head derived from the target head with the correct bit width;
- correct Hadamard/`suh` transforms for fused QKV, gate/up, and GDN-related shards;
- correct group metadata and `shard_of_nb` handling;
- exact expected tensor shapes;
- no assumption of tensor-parallel support unless implemented and validated.

### Implementation tasks

- [ ] Add an explicit loader validation report emitted after weight load.
- [ ] Compare expected vs. loaded:
  - tensor names,
  - shapes,
  - quantization bit widths,
  - groups,
  - shard identities,
  - MTP presence.
- [ ] Fail startup on missing/unexpected quantized weights unless explicitly permitted.
- [ ] Add representative reconstruction tests against an independent reference path.

### Pass criteria

No loader mismatch, silent tensor omission, bit-width mismatch, or fused-shard mismatch.

---

## 7. Phase 3 — Align the EXL3 serving contract with production before claiming capacity gains

### Current mismatch to eliminate

The reviewed profiles differ materially:

| Setting | GPTQ production | Existing EXL3 profile |
|---|---:|---:|
| Runtime | vLLM 0.30 / Torch 2.13 | vLLM 0.26.1 / Torch 2.12 |
| Configured context | 200,704 | 262,144 |
| Max sequences | 4 | 16 |
| Prefill token budget | 6,656 | 4,096 |
| GPU memory utilization | 0.93 | 0.965 |
| MTP depth | 4 | 3 |
| Draft vocabulary | full | 65,536 rows |
| Media limit | 1 image / 0 video | 32 images / 4 videos |

These differences make direct capacity conclusions unsafe.

### Initial target contract

Before changing the runtime version, configure the existing EXL3 image as closely as possible to the actual production contract:

- [ ] total context limit: **200,704 tokens**;
- [ ] concurrency target: **C4**;
- [ ] production media limit;
- [ ] production API/tool/reasoning parser behavior;
- [ ] FP8 KV behavior;
- [ ] prefix cache behavior;
- [ ] equivalent admission/scheduler semantics where supported.

Do not increase memory utilization merely to force a pass.

### API validation

- [ ] Verify health endpoint.
- [ ] Verify normal text generation.
- [ ] Verify streaming.
- [ ] Verify tool-call parser behavior used by production.
- [ ] Verify reasoning parser behavior used by production.
- [ ] Verify image input under the allowed production limit.
- [ ] Verify over-limit requests fail in the expected way.
- [ ] Verify cancellation/abort.
- [ ] Verify prefix reuse.

### Pass criteria

The EXL3 candidate implements the same externally relevant serving contract needed by production before its capacity/performance results are compared with GPTQ.

---

## 8. Phase 4 — Diagnose long-context preemptions before optimizing throughput

### Why this is a release blocker

The supplied EXL3 runs preempt at approximately 139K and 188K input tokens, while the corresponding GPTQ runs do not.

This must be explained before EXL3 can replace the existing long-context production promise.

Do not assume the cause is a transient oneDNN OOM. Do not assume the entire prompt was recomputed merely because a scheduler preemption occurred.

### Instrumentation to add

Record memory/cache state at least at:

1. after model weights load;
2. after MTP/draft structures are created;
3. after graph capture;
4. after warm-up;
5. immediately before long-context request admission;
6. immediately before cache slot allocation;
7. immediately before a preemption decision;
8. after preemption/requeue;
9. after request completion/abort.

Record:

- [ ] device allocated memory;
- [ ] device reserved memory;
- [ ] free memory if reliably available;
- [ ] graph-pool memory where available;
- [ ] oneDNN/workspace allocations where observable;
- [ ] KV cache groups and block sizes;
- [ ] requested blocks by cache group;
- [ ] available blocks by cache group;
- [ ] prefix-cache references;
- [ ] lookahead/speculative reservations;
- [ ] GDN state dtype and block usage;
- [ ] actual EXL3 KV page size/layout;
- [ ] scheduler reason for preemption;
- [ ] whether prefix blocks are evicted;
- [ ] whether logical prefill counters represent new compute or request-length accounting.

### Required reproductions

Run from a fresh worker for each key case:

- [ ] ~103K input, C1;
- [ ] ~139K input, C1;
- [ ] ~188K input, C1;
- [ ] production total-context boundary near 200,704 tokens;
- [ ] prefix extension case;
- [ ] abort/restart case;
- [ ] C4 reference load;
- [ ] one allowed image under long context if multimodal production traffic requires it.

### Decision logic

If preemption is caused by a real cache-capacity deficit:

- quantify the deficit by cache group;
- identify whether it originates from MTP, GDN, KV padding, graphs, scheduler lookahead, or another pool;
- reduce/reshape the responsible allocation before raising the global memory-utilization target.

If preemption is caused by scheduler/admission accounting:

- correct the accounting/contract;
- add a regression test around the exact boundary.

### Pass criteria

- No unexplained preemption at the production long-context boundary under the required C1/C4 contract.
- No unbounded recomputation behavior.
- No OOM.
- Stable, explainable memory/cache accounting.

---

## 9. Phase 5 — Port EXL3 cleanly to the target vLLM 0.30 / Torch 2.13 runtime

Only start this after the existing EXL3 image is understood under the production contract. This prevents runtime-port bugs from being confused with checkpoint or scheduler issues.

### 9.1 Treat installed target-runtime code as authoritative

Do not blindly copy all old EXL3 monkey patches.

Re-evaluate each patch against the actual target vLLM 0.30 source.

### 9.2 Do not reapply the old 64-alignment block-size override if the target runtime already contains it

The reviewed vLLM 0.30 XPU platform code already rounds the relevant block size to a multiple of 64 and adjusts the Mamba/padded page size accordingly.

Tasks:

- [ ] remove or disable the redundant old EXL3 override;
- [ ] log the resulting actual page/block sizes at startup;
- [ ] add a regression test for expected alignment.

Do **not** hard-code GPTQ's observed `1664` page size into EXL3. The correct EXL3 value must come from its actual runtime layout.

### 9.3 Replace the broad/partial GDN regex patch with a semantic target-runtime implementation

Relevant legacy patch area:

`exl3-engine/exl3xpu/vllm_patches.py`

The static review found six replacements against the supplied 0.30 source, while additional sites using `non_spec_sequence_masks_cpu` remained, including GPU indexing of block tables/query lengths.

Tasks:

- [ ] identify every synchronization-producing CPU/GPU mask/index conversion in the active 0.30 GDN path;
- [ ] prepare CPU indices once where possible;
- [ ] reuse already available CPU metadata instead of calling `.item()` or re-materializing masks inside layer execution;
- [ ] patch the active XPU/V2 runner path, not a legacy runner class that is not actually used;
- [ ] add semantic tests for:
  - all-spec,
  - all-non-spec,
  - mixed spec/non-spec,
  - C1,
  - C4,
  - graph capture/eager.

Pass only when every intended synchronization site is accounted for. Do not treat replacement count as proof.

### 9.4 Validate plugin registration in every worker

- [ ] Ensure out-of-tree EXL3 quantization registration occurs before model construction in every worker process.
- [ ] Fail startup if the quantization method is unavailable.
- [ ] Log plugin version/source hash once per worker.

### 9.5 ABI and native-library requirements

- [ ] Recompile native code against the target Torch 2.13 environment.
- [ ] Do not copy a Torch 2.12-built `.so` into the target runtime.
- [ ] Run import/op-registration smoke tests before model load.

### Port pass criteria

- Model loads cleanly on the target runtime.
- All expected EXL3 weights are consumed.
- MTP works without GPTQ-specific overrides.
- API contract passes.
- No hidden CPU-sync regression is introduced by stale patches.
- Baseline quality/correctness checks remain within tolerance before any performance tuning.

---

## 10. Phase 6 — Revalidate numerical correctness and quality on the production precision path

### 10.1 Reuse existing BF16 reference data

Do not rebuild the full reference corpus unnecessarily.

Use the existing quantization reference panel and add new EXL3 arms.

Suggested arms:

1. existing known EXL3 runtime/checkpoint;
2. EXL3 on the new vLLM 0.30 / Torch 2.13 port;
3. EXL3 with production FP8 KV enabled;
4. EXL3 with production INT8 prefill enabled;
5. EXL3 with production graph path enabled;
6. EXL3 with production MTP configuration;
7. full candidate production path.

Where practical, enable one new component at a time so quality regressions can be localized.

### 10.2 Preserve the validated scoring methodology

Keep:

- identical token IDs across arms;
- correct next-token alignment;
- exclusion of the last unscoreable logit;
- full-vocabulary distributions for KL positions;
- FP64 normalization for divergence computation if that is the established implementation;
- perplexity computed as `exp(mean NLL)`, not mean window perplexity;
- bootstrap at the window level, not treating every token as independently sampled.

### 10.3 Add long-context quality probes

The current quality panel is short-context.

Add a small number of fixed long contexts, for example representative contexts near:

- 32K,
- 100K,
- 180K,
- production maximum where practical.

For each, score a short fixed continuation suffix against the same BF16/reference target where feasible.

This is not intended to become a huge benchmark campaign; it is a regression detector for long-context precision-path failures.

### 10.4 Verify vocabulary/padding semantics

- [ ] Confirm the actual tokenizer vocabulary size.
- [ ] Confirm handling of padded vocabulary rows.
- [ ] Confirm masking is identical between compared arms.
- [ ] Check probability mass assigned outside the valid tokenizer vocabulary when applicable.
- [ ] Never mask only one candidate arm differently.

### Quality pass criteria

- The port does not materially erase the measured EXL3 advantage over GPTQ.
- No new numerical failures or top-level functional regressions appear.
- Long-context precision-path checks remain stable.
- Any accepted quality delta is explicitly documented with raw metrics.

Do not define a new arbitrary pass threshold after seeing the results. If a hard threshold is desired, commit it before the final run.

---

## 11. Phase 7 — Decode optimization priority 1: MTP3 vs. MTP4

This is the first performance experiment.

### Hypothesis

A significant portion of the clean 103K decode gap may be due to the different speculative-decoding depth and resulting accepted/generated tokens per round, rather than solely slower EXL3 target linears.

### Experimental matrix

Test EXL3 with at least:

- MTP3, C1;
- MTP4, C1;
- MTP3, C4;
- MTP4, C4.

Use fixed prompts and fixed output-token budgets across variants.

Include at least a small representative context set, such as:

- 4K,
- 32K,
- ~103K clean-comparison prompt,
- ~128K or another long context that does not preempt after Phase 4 is fixed.

Use both code-oriented and prose-oriented prompts if acceptance behavior differs meaningfully by workload.

### Metrics to record

Per request / aggregate as appropriate:

- [ ] TTFT;
- [ ] prefill tok/s;
- [ ] decode tok/s;
- [ ] output tok/s aggregated for C4;
- [ ] number of speculative rounds;
- [ ] draft tokens proposed;
- [ ] draft tokens accepted;
- [ ] acceptance by speculative position if available;
- [ ] generated tokens per round;
- [ ] full speculative cycle latency;
- [ ] target-model latency;
- [ ] draft-model latency;
- [ ] LM-head latency;
- [ ] sampler/commit latency;
- [ ] graph bucket / actual row count / padded row count;
- [ ] p50/p95/max inter-token gap;
- [ ] preemptions;
- [ ] energy or joules/output token if available and stable enough to measure.

### Important constraints

- Do not assume MTP4 is better because it produces more draft positions.
- Previous EXL3 development evidence indicates MTP4 can help C1 and hurt C4 depending on graph/matrix behavior.
- Do not reintroduce previously unstable dynamic-k behavior unless there is a new design that respects graph resource constraints.
- Do not change the draft-block allocation simultaneously with the MTP-depth experiment unless necessary for correctness.

### Pass/selection rule

Choose the MTP depth per serving profile based on measured end-to-end behavior, not draft acceptance alone.

If C1 and C4 prefer different configurations, support explicit profile selection rather than forcing one universal value.

---

## 12. Phase 8 — Decode optimization priority 2: port Shared-KV Verification (M04) carefully

Relevant GPTQ/reference areas:

- installed `b70_attention_base.py`
- `gptq-engine/b70_ops/csrc/shared_kv_verification.cpp`
- `gptq-engine/b70_ops/patches/shared-kv-verification.patch`

### Why it is promising

This optimization operates in attention/verification rather than the weight quantization format itself. Therefore the concept can be portable from GPTQ to EXL3.

It is especially relevant when long-context verification attention is a significant fraction of each speculative cycle.

### Current contract that must not be copied blindly

The reviewed M04 implementation is specialized for approximately:

- one sequence;
- 2–5 verification query rows;
- FP16 Q;
- FP8 E4M3 K/V;
- 24 query heads / 4 KV heads;
- head dimension 256;
- the current GPTQ page/layout assumptions, including a reviewed `1664` page-size dependency.

It also relies on a special causal-mask rule for multiple verification rows:

For query row `r`, visible keys must stop at the position equivalent to `L - q + r`.

Copying only the wrapper without preserving this causal behavior is incorrect.

### Implementation tasks

- [ ] Profile the EXL3 speculative cycle first to confirm verification attention is material.
- [ ] Parameterize or rebuild the kernel for the actual EXL3 KV page/layout/strides.
- [ ] Do not hard-code GPTQ page size.
- [ ] Preserve exact FP8 scale semantics.
- [ ] Preserve causal visibility for every verification row.
- [ ] Ensure KV update is performed exactly once.
- [ ] Add strict shape/dtype/layout guards.
- [ ] Unsupported shapes must fall back to the native path without changing results.
- [ ] Validate q lengths at least around:
  - 1,
  - 2,
  - 3,
  - 4,
  - 5,
  - larger unsupported values for fallback behavior.
- [ ] Validate exact KV lengths around page boundaries.
- [ ] Validate graph capture and eager execution.

### C4 requirement

Do not simply call the single-sequence M04 kernel four times and assume C4 will improve.

Profile that approach. If host-launch overhead erases the gain, investigate a genuinely batched verification kernel or a dispatcher that groups compatible work.

### Pass criteria

- Bit/numerical agreement within the predefined attention tolerance.
- Correct causal masking.
- No incorrect behavior at page boundaries.
- No regression on unsupported shapes.
- Measurable end-to-end speculative-cycle benefit, not only a microbenchmark win.

---

## 13. Phase 9 — Decode/mixed-serving priority 3: unified, guarded attention dispatch

The production goal is not simply “turn on oneDNN everywhere.” EXL3 already has a oneDNN prefill path.

The useful work is to dispatch by **actual request shape and validated contract**.

### Desired routing model

| Actual workload | Preferred path |
|---|---|
| Long prefill | oneDNN only when exact supported KV length/layout is satisfied |
| Short prefill | existing Q128/native path according to its validated contract |
| Pure decode / speculative verify | native or M04-derived path when supported |
| Mixed prefill + decode | correctly split metadata and appropriate subpaths |
| Unsupported shape/dtype/layout | unchanged native fallback |

### Requirements

- [ ] No Eager-only oneDNN call may be accidentally captured into an invalid graph path.
- [ ] Do not silently ignore sliding-window semantics, ALiBi, attention sinks, special masks, or alternate output formats if any are active.
- [ ] Do not update KV twice when splitting work.
- [ ] Use the actual active vLLM 0.30 XPU/V2 runner interfaces.

### Legacy mixed-attention issues to fix rather than copy

Relevant legacy area:

`exl3-engine/exl3xpu/vllm_patches.py`

The reviewed legacy code depends on `num_prefill_reqs` / `num_decode_reqs` metadata that was not available in the older measured path, and contains a `max_query_len` construction based on total decode rows rather than the true maximum per sequence.

It also performs a per-layer `.item()` on sequence length in the reviewed path.

### Implementation tasks

- [ ] Define an explicit metadata contract at runner level.
- [ ] Precompute needed CPU metadata once per step.
- [ ] Use true per-sequence maxima.
- [ ] Remove avoidable layer-level host synchronization.
- [ ] Test C1 and C4 mixed workloads.
- [ ] Include a critical production-style test: an already-decoding request while a large cold prompt begins prefill.

### Primary metric

For mixed serving, focus not only on average decode tok/s but also on **p95/max token gaps for existing decode requests during a large incoming prefill**.

---

## 14. Phase 10 — Remove the avoidable host scalar read in EXL3 oneDNN prefill

Relevant areas:

- `exl3-engine/csrc/exl3_ops.sycl`
- `exl3-engine/exl3xpu/fp8kv_prefill.py`

### Reviewed issue

The oneDNN prefill native path performs a host read equivalent to:

```cpp
st.item<float>()
```

to make a host-side decision, even though the relevant scale is effectively constant for the model configuration. The Python layer also invokes the operation separately for KV heads.

### Implementation

- [ ] Move stable scale selection into host-side configuration/cache state.
- [ ] Keep the appropriate device scalar/tensor available without rereading the old device value on every invocation.
- [ ] Ensure correct device, stream, and lifetime semantics.
- [ ] Verify thread/worker safety.
- [ ] Profile before/after under:
  - long prefill,
  - mixed prefill+decode,
  - C1,
  - C4.

### Scope warning

This is primarily a **prefill/mixed-serving synchronization optimization**, not a direct Small-M pure-decode optimization. Do not present it as the solution to the entire decode gap unless measurement proves otherwise.

---

## 15. Phase 11 — Keep the existing EXL3 row dispatch; port GPTQ's tests, not its implementation

### Reviewed finding

GPTQ required a row-dispatch repair because the compiler/graph path could freeze the large W4A8 branch into small decode shapes.

EXL3 already performs row-count dispatch inside an opaque native operation, with the Small-M path returning before the INT8-prefill branch.

Therefore a second GPTQ-style row dispatcher is not currently justified.

### What to port

Port the **regression methodology**:

- [ ] trace/capture a large shape first, then run a small shape;
- [ ] test M = 128 and M = 129 boundary behavior;
- [ ] test actual vs. padded rows;
- [ ] test Graph and Eager;
- [ ] test mixed matrices/shapes in realistic sequences;
- [ ] prove that padded rows do not affect valid outputs.

### Shape coverage

At minimum include relevant classes such as:

`M = 1, 2, 3, 4, 5, 8, 12, 16, 20, 24, 32, 40, 48, 64, 128, 129`

plus representative large-prefill shapes.

### Pass criteria

The correct kernel family is selected from the actual current row count under every tested capture/order scenario.

---

## 16. Phase 12 — Kernel tuning only after the new end-to-end profile

Relevant EXL3 areas:

- `exl3-engine/csrc/exl3_esimd.h`
- `exl3-engine/csrc/exl3_ops.sycl`
- `exl3-engine/docs/PROGRESS.md`

### First profile the final candidate

After Phases 7–11, collect a fresh profile at the target power/runtime settings.

Break the speculative cycle into at least:

- target EXL3 linears;
- target LM head;
- attention/verification;
- GDN/state work;
- draft/MTP work;
- sampling;
- commit/scatter;
- host synchronization;
- launch/graph overhead.

Only optimize components that are still material.

### Allowed targeted tuning

If target EXL3 linears remain dominant, perform narrow, shape-weighted experiments around:

- NT/workgroup choices;
- MB/tile choices;
- split-K thresholds;
- M=1/2 vector path;
- M=3/4 and small-M DPAS/XMX crossover;
- 6 bpw target-head specific tuning;
- compiler-version-sensitive scheduling.

Use the actual production shape histogram instead of broad synthetic sweeps.

### Previously explored directions not to repeat without a new hypothesis

Do not spend time re-running these unchanged approaches merely because they are easy to try:

- global 65,536-entry codebook LUT: previously severe M1 regression;
- blanket Hadamard fusion under graphs;
- register-heavy double-buffer prefetch;
- treating Vector M3/M4 as universally superior;
- unchecked `EXL3_K4_HALVES`, which produced incorrect results;
- `EXL3_FOLD` as a free optimization, because it changes weight rounding/numerical behavior;
- alignment-sync skip without a new synchronization design;
- padding K length merely to reduce oneDNN primitive variants when that padding is known to produce wrong results.

### Amdahl constraint

A historical EXL3 profile attributed roughly 31.7 ms of a 41 ms cycle to target linears. That profile is not the final 180 W target measurement, but it illustrates the limit: even a 20% improvement in the linear portion would not automatically close a ~40% end-to-end throughput requirement.

Optimize the measured critical path, not the most interesting kernel in isolation.

---

## 17. Attention correctness matrix

Any new attention dispatcher/M04/mixed path must include a compact but strong correctness matrix.

### Query lengths

Test at least:

- q = 1;
- q = 2–5;
- q = 63/64;
- q = 255/256 where supported or relevant;
- unsupported sizes to confirm fallback.

### KV lengths

Use exact KV lengths around:

- page boundaries;
- route thresholds;
- production-long-context regions.

Do not replace exact K length with unsafe padding just to hit a cached primitive.

### Modes

- [ ] C1
- [ ] C4
- [ ] prefill only
- [ ] decode only
- [ ] speculative verification
- [ ] mixed prefill/decode
- [ ] Eager
- [ ] graph capture/replay

### Correctness assertions

Verify:

- causal visibility;
- FP8 scales;
- Q/K/V strides;
- page indexing;
- valid output rows only;
- fallback equivalence;
- no duplicate KV writes;
- no NaNs/Inf.

---

## 18. Performance benchmark matrix

Keep the benchmark suite intentionally small and reproducible rather than turning it into an open-ended campaign.

### 18.1 Controlled serving points

At minimum benchmark:

- 4K input;
- 32K input;
- the existing ~103K clean-comparison prompt;
- ~128K or another long context known to be non-preempting;
- near the 200,704 production boundary after capacity is fixed.

### 18.2 Concurrency

Run:

- C1;
- C4.

### 18.3 Repetition strategy

- Alternate variant order to reduce thermal/drift bias.
- Warm the intended graph/primitive caches consistently.
- Preserve a cold-start benchmark separately from steady-state results.
- Record any run with preemption and exclude it from a “clean kernel/throughput” comparison while retaining it as operational evidence.

### 18.4 Metrics

Report:

- TTFT;
- prefill tok/s;
- decode tok/s;
- aggregate output tok/s;
- p50/p95/max token gap;
- speculative acceptance metrics;
- cycle latency;
- preemptions;
- peak/reserved device memory;
- cache-block usage;
- energy/output token if sufficiently stable;
- failure/timeout count.

### 18.5 Mixed-load benchmark

Include at least one scenario where:

1. one or more requests are already decoding;
2. a large cold prompt begins prefill;
3. token-gap impact on the existing requests is measured.

This is necessary because a dispatcher can improve average throughput while making interactive tail latency worse.

---

## 19. Product-quality / coding-task validation

The existing FlappyBird/coding evidence is useful but should not be treated as an isolated model-quality ranking.

### Known limitations of the old run

- EXL3 and GPTQ traversed different adaptive trajectories.
- Context lengths and repair requests differed.
- A visible storage test was too strict and penalized a correctly lazy-loading GPTQ solution during the live run.
- Post-hoc rescoring cannot undo the effect that test had on the original trajectory.
- The remaining GPTQ failures after correction relate to the requested `data-testid`, not a totally missing input field.

### Validation plan

- [ ] Use the corrected v7 task/test version.
- [ ] Run a small paired comparison with equal budgets.
- [ ] Add one independent representative coding/agent task.
- [ ] Preserve tool/API configuration across engines.
- [ ] Record task success, retries, tokens, wall-clock time, and major failure categories.

### Interpretation

Use this as practical utility evidence alongside PPL/KL and serving metrics. Do not let one adaptive task override deterministic correctness/capacity failures.

---

## 20. Production release gates

EXL3 may replace GPTQ only after all mandatory gates below are explicitly marked PASS.

### Gate A — Artifact/build integrity

- [ ] Native build fails closed.
- [ ] New `.so` hash and build manifest recorded.
- [ ] Correct ABI/runtime linkage verified.
- [ ] Plugin loads in every worker.
- [ ] No stale binary can masquerade as a successful rebuild.

### Gate B — Loader/tensor correctness

- [ ] Full weight-index Trellis enumeration covered by tests.
- [ ] Correct 4 bpw / 6 bpw assignments.
- [ ] Fused shard transforms validated.
- [ ] MTP tensors covered.
- [ ] No missing/unexpected tensor.

### Gate C — Kernel/shape correctness

- [ ] Small-M and prefill dispatch boundaries pass.
- [ ] Graph/Eager pass.
- [ ] Large-first capture regression passes.
- [ ] No NaN/Inf.
- [ ] Padded rows do not alter valid output.

### Gate D — Attention correctness

- [ ] Exact causal behavior verified.
- [ ] Page-boundary tests pass.
- [ ] Mixed C1/C4 tests pass.
- [ ] Unsupported shapes fall back safely.

### Gate E — Quality

- [ ] Existing reference panel rerun for the port/candidate arms.
- [ ] EXL3 quality advantage is not materially destroyed by the production precision path.
- [ ] Long-context suffix probes show no unexplained regression.
- [ ] No new tool/parser/application correctness failures.

### Gate F — Capacity / operational stability

- [ ] 139K and 188K cases explained and stable.
- [ ] Production 200,704 total-context contract passes.
- [ ] Required C4 workload passes.
- [ ] Prefix extension passes.
- [ ] Abort/restart passes.
- [ ] Production media limit passes.
- [ ] No unexplained preemption/recompute/OOM.

### Gate G — Performance

- [ ] MTP3 vs. MTP4 decision is based on controlled results.
- [ ] C1 performance measured.
- [ ] C4 aggregate throughput measured.
- [ ] Tail token-gap behavior measured under mixed load.
- [ ] Long-context clean pair repeated.
- [ ] Any remaining decode regression is explicitly accepted before release.

### Gate H — Rollback

- [ ] Existing GPTQ image remains deployable.
- [ ] Rollback command/procedure is documented and tested.
- [ ] No shared incompatible runtime caches can survive rollback/upgrade boundaries.

---

## 21. Suggested stop conditions

Stop the migration and keep GPTQ production if any of the following cannot be resolved within the agreed engineering budget:

- unexplained missing/incorrect EXL3 tensors;
- persistent attention or loader correctness errors;
- NaNs/Inf;
- material loss of the measured EXL3 quality advantage in the real production precision path;
- inability to meet the current long-context/C4 contract;
- recurring unexplained preemptions or recomputation;
- unacceptable tail latency/token stalls;
- decode regression that exceeds the product's agreed quality-vs-latency tradeoff;
- build/runtime instability that cannot be made reproducible.

Do not use lower PPL alone as a reason to waive operational correctness.

---

## 22. Codex implementation sequence

Codex should execute the work in this order and avoid jumping ahead unless a dependency requires it.

### Milestone 1 — Safe foundation

- [ ] Baseline manifest.
- [ ] Build script fail-closed repair.
- [ ] EXL3/GPTQ MTP override isolation.
- [ ] Bit-exact tensor enumeration repair.
- [ ] INT8-prefill test proper exit status.

**Commit boundary:** no performance changes yet.

### Milestone 2 — Existing EXL3 image under production contract

- [ ] Align API/context/concurrency/media settings.
- [ ] Add cache/memory/preemption instrumentation.
- [ ] Reproduce 103K/139K/188K and production-boundary cases.
- [ ] Explain and eliminate unacceptable preemption behavior.

**Commit boundary:** operationally understood existing EXL3 stack.

### Milestone 3 — Runtime port

- [ ] Rebuild for vLLM 0.30 / Torch 2.13.
- [ ] Remove obsolete block-size patch.
- [ ] Replace partial GDN regex behavior with target-runtime implementation.
- [ ] Validate plugin/loader/MTP/API.

**Commit boundary:** functionally correct port before optimization.

### Milestone 4 — Quality qualification

- [ ] Reuse existing BF16 panel.
- [ ] Compare old EXL3 vs. ported EXL3 vs. staged production precision path.
- [ ] Add small long-context suffix checks.

**Commit boundary:** known numerical/quality baseline for the new runtime.

### Milestone 5 — Highest-impact decode work

- [ ] Controlled MTP3/MTP4 matrix for C1/C4.
- [ ] Instrument full speculative-cycle breakdown.
- [ ] Select profile(s) based on end-to-end results.

**Commit boundary:** speculative configuration fixed before lower-level kernel work.

### Milestone 6 — Attention and mixed-serving optimization

- [ ] Port/parameterize M04 only where profile supports it.
- [ ] Build guarded attention dispatcher.
- [ ] Fix mixed metadata/CPU-sync issues.
- [ ] Remove repeated scale host read.
- [ ] Validate mixed-load token gaps.

**Commit boundary:** attention improvements independently measurable and revertible.

### Milestone 7 — Targeted EXL3 kernel tuning

- [ ] Fresh production-shape profile.
- [ ] Only tune dominant shapes/components.
- [ ] Keep every experiment correctness-gated.

**Commit boundary:** one optimization family per commit where practical.

### Milestone 8 — Release candidate

- [ ] Run compact release matrix.
- [ ] Run corrected coding/task pair.
- [ ] Produce final comparison report.
- [ ] Verify rollback.
- [ ] Make explicit release decision based on pre-agreed gates.

---

## 23. Required repository artifacts

Codex should leave enough evidence for another engineer to audit what changed and why.

Recommended files/directories:

```text
docs/
  EXL3_PRODUCTION_MIGRATION_AND_OPTIMIZATION_PLAN.md
  EXL3_PORT_NOTES.md
  EXL3_RELEASE_REPORT.md

benchmarks/results/exl3-migration/
  baseline_environment.json
  capacity_*.json
  mtp3_vs_mtp4_*.json
  attention_*.json
  quality_*.json
  mixed_load_*.json
  release_summary.json

build/
  exl3_build_manifest.json
```

The exact locations may follow existing repository conventions, but keep raw machine-readable results separate from prose summaries.

### Every benchmark result should record

- source commit;
- dirty-tree status;
- runtime image;
- native library hash;
- model/checkpoint hash;
- launch command/arguments;
- environment variables relevant to kernels/scheduling;
- B70 driver/runtime information;
- power profile;
- benchmark input identifier/hash;
- concurrency;
- context length;
- output budget;
- MTP depth;
- whether graphs are enabled;
- whether preemption occurred;
- timestamps;
- raw metrics.

---

## 24. Required final release report structure

`docs/EXL3_RELEASE_REPORT.md` should contain:

1. **Executive summary** — whether EXL3 is ready to replace GPTQ under the defined production contract.
2. **Exact artifacts tested** — commits, images, model hashes, native-library hashes.
3. **Changes implemented** — grouped by build, loader, runtime port, attention, MTP, kernels.
4. **Correctness results** — tensor, kernel, attention, API.
5. **Quality results** — PPL, NLL, KL, long-context checks, task validation.
6. **Capacity results** — C1/C4, long-context boundary, preemption diagnosis.
7. **Performance results** — TTFT, prefill, decode, aggregate throughput, token gaps, MTP acceptance/cycles.
8. **Regression analysis** — anything worse than GPTQ and why.
9. **Known limitations** — unsupported shapes/features/TP/etc.
10. **Rollback procedure**.
11. **Release decision and the exact gates used**.

Do not delete unsuccessful results. Keep them as evidence, clearly marked as failed experiments.

---

## 25. Optimization hypotheses ranked by priority

This is the recommended order based on the reviewed evidence, not a claim that every item will improve performance.

| Priority | Hypothesis | Why | Main target |
|---|---|---|---|
| P0 | Fix build/loader/test safety | Required before trusting any measurement | correctness/reproducibility |
| P0 | Diagnose long-context preemptions | Production release blocker | capacity/stability |
| P1 | MTP3 vs. MTP4 | Existing 103K counters show large tokens/round difference | decode throughput |
| P2 | Shared-KV verification / M04 | Quantization-independent, potentially important at long context | verification decode |
| P3 | Guarded attention dispatcher + mixed metadata | Can reduce stalls under mixed serving | tail latency / throughput |
| P3 | Remove repeated `st.item<float>()` scale read | Concrete avoidable host sync | prefill/mixed |
| P4 | Narrow EXL3 kernel retuning on new compiler/profile | Useful only if linears remain dominant | target linear latency |
| P4 | LM-head/sampler/commit tuning | Only if fresh profile shows material cost | cycle overhead |

A performance patch should be retained only if its end-to-end benefit survives the corresponding correctness and mixed-load tests.

---

## 26. Final decision framework

At the end of this plan, compare the final EXL3 release candidate against the pinned GPTQ production baseline on four independent axes:

### Quality

Does EXL3 retain the materially lower divergence / lower perplexity degradation observed in the current evidence under the real production precision path?

### Capacity and stability

Can EXL3 meet the actual 200,704-token/C4 production contract without unexplained preemption, recomputation, or OOM, and does the smaller weight footprint produce useful additional operating margin?

### Latency and throughput

After MTP/attention/runtime optimization, what decode-throughput and tail-latency regression remains relative to GPTQ at C1, C4, and long context?

### Product utility

On corrected paired coding/agent tasks, does the higher quantization fidelity translate into useful practical behavior without unacceptable serving cost?

The production migration should be made only from this combined evidence. A remaining latency regression can be an intentional product tradeoff if it is explicitly accepted in exchange for quality and/or usable context headroom, but it should be documented rather than hidden behind an adaptive task result.

---

## 27. Expected outcome

The most likely productive path is:

1. keep GPTQ as the current rollback-safe production engine;
2. qualify EXL3 as the higher-fidelity candidate;
3. fix the release-blocking build/test/runtime issues;
4. resolve long-context preemptions;
5. port EXL3 cleanly to the production runtime;
6. optimize MTP behavior first;
7. then reuse the quantization-independent GPTQ attention work where contracts genuinely match;
8. perform new Trellis-kernel tuning only after the updated profile shows it is still the limiting factor.

The key engineering principle is to avoid treating the current ~28.5% clean decode gap as a single “EXL3 kernel problem.” The supplied evidence indicates that speculative-round productivity, attention/verification cost, scheduler/capacity behavior, runtime differences, and target-linear latency all need to be separated before deciding where the remaining performance budget can actually be recovered.


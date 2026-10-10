# B70 EXL3 / MTP3: implementation and experiment plan

**Review date:** 2026-10-10  
**Input:** `b70-stack-patches-pro-review-2026-10-10.zip`  
**Primary goal:** higher useful decode throughput; preserve prefill, long-context capacity and concurrent serving.  
**Decision already made:** keep native MTP3. DFlash and speculative-depth sweeps are out of scope.

## 0. Start here: the recommended decision

Do not begin with a general dependency upgrade or an entire upstream optimization bundle. First make a small, read-only route/latency inventory of the qualified image, and check the oneDNN source-to-binary provenance. Then test one isolated change at a time.

The first dependency candidate is **oneDNN 3.13.4, primarily for correctness/maintenance**, not an asserted decode speedup. The first decode kernel candidates are **GDN gated RMSNorm** and **QKV/norm/RoPE**, provisionally in that order. Their order must follow measured *captured GPU time*, not upstream per-call headlines. A material M04 or EXL3 bottleneck may move ahead of either candidate.

There is no evidence here that upgrading Torch alone will recover the 64K regression. Investigate that regression through a bounded same-prefix/state comparison, separately from kernel timing. Do not reopen the full old qualification or MTP-depth matrix.

### Ordered decision queue

| ID | Priority | One logical change / question | Gate before GPU serving experiments |
|---|---|---|---|
| P0 | First; diagnostic only | Pin the live route, timing, memory and original negative findings | Confirm that each proposed optimization reaches production graphs; emit a component budget |
| D64 | Small diagnostic alongside P0 | Explain 64K acceptance/trajectory change on identical inputs | Existing saved references first; no blanket old-image rerun |
| E01 | First dependency candidate | Custom oneDNN 3.13.0 → 3.13.4, otherwise unchanged | Exact source/binary mapping, relevant SDPA implementation, import/ABI and numerical gates |
| E02 | First provisional decode patch | Minimal FP16 GDN gated-RMSNorm kernel plus actual rank-3 graph dispatch | P0 proves a meaningful active norm cost and a graph-reachable route |
| E03 | Next provisional decode patch | SYCL QKV/norm/RoPE instead of the already fused Triton implementation | Actual FP16 layout, MRoPE/rounding parity and C4 row coverage |
| E04a | Move earlier if attention reduction dominates | Port only head-dimension reduction tiling to the separately built M04 | Nontrivial reduction share, original q2–5 masks and scratch contract preserved |
| E04b | Separate optional candidate | M04 reduction writes directly into the native output layout | Material copy/launch cost and a proven output-index mapping |
| E05 | Conditional high-value custom work | One measured EXL3 producer/consumer boundary, not a new mega-kernel | New opportunity beyond historically rejected MB4/split-K/full-fusion trials |
| E06 | Only with measured device-idle time | One metadata/host/graph bottleneck | Timeline proves idle time is avoidable host work, not a wait for GPU completion |
| E07 | Later maintenance | Torch 2.14.1 with compatible rebuilt extensions | Clean dependency resolution and explicit ABI/source-guard updates |
| HOLD | Not now | Loader, driver, compiler or nightly updates | A specific applicable fix or a concrete reproduction must justify each |

The queue is not an instruction to implement every item. Stop an individual candidate when its route is inactive, its achievable benefit is too small, or a required invariant fails. Save the negative result and move on.

## 1. Baseline and evidence boundaries

All archive paths below are relative to its top-level directory. Abbreviations:

- **V** = `runtime/qualified/installed-vllm`
- **E** = `exl3-qualified-source`
- **R** = `repository`
- **P** = `research/prs`

The actual installed sources and runtime receipts are authoritative. An experimental directory or an upstream PR being present does not mean it is deployed.

### Immutable baseline

| Item | Keep fixed unless it is the named candidate variable |
|---|---|
| Production image | `sha256:194a8786f12065e65647d914ddc7fdc136bafbeec73f476730ef1324e4d2c7ec` |
| Owned repository | `260dd4f900c4662386aff8a80413d09d9465202e` |
| EXL3 source | `fc8f8c9fa1627f28a7fbe6ee4fc7bf961dabef81` |
| M04 kernel-header base | `6d92b1bfbf32767ecda8e819613eb151e70030ad` |
| SYCL TLA | `87f6850680a580654b9ea2c80dbc01aeb36ad231` |
| Hardware/power | Intel Arc Pro B70, 32 GB; 180 W |
| Model | Same Qwen3.8-27B EXL3 4.00-bpw checkpoint and optimized recipe, hashes unchanged |
| Arithmetic | FP16 model activations; mathematically necessary internal FP32 accumulation allowed; no BF16 bridge |
| Speculation/cache | Native MTP3; FP8 target KV; prefix caching enabled; Mamba cache mode `align` |
| Context/admission | 262144 total input-plus-output tokens; `max_num_seqs=16` |
| Scheduler | `max_num_batched_tokens=4096`; full-input reservation enabled; watermark 0 |
| Memory | `gpu_memory_utilization=0.965`; no hidden reduction of usable KV capacity |
| Graphs | `FULL_DECODE_ONLY`; sizes `[1,2,4,8,12,16,24,32,40,48,56,64]` |
| Stack | vLLM 0.31.0+xpu distribution; runtime `vllm.__version__ == '0.31.0'`; Torch 2.14.0+xpu; Triton XPU 3.8.0; vllm-xpu-kernels 0.1.15.4 |
| Native toolchain | oneAPI/SYCL runtime 2026.1.1, SYCL 9 family; C++20; Torch CXX11 ABI true |
| Custom oneDNN | 3.13.0, reported Git `0e2a5bfeef1bfbffc3137464606540233086ce9b`; runtime `/opt/b70/libdnnl.so.3` |
| Active attention | Original qualified M04 q2–5, externally gated to C1–C4; guarded oneDNN prefill |
| oneDNN partition cache | Capacity 64; exact K length; validated Q padding |

Sources: `R/config/production_image.json`, `SOURCE_PROVENANCE.json`, `runtime/qualified/CURRENT_LIBRARY_VERSIONS.json`, `runtime/qualified/exl3xpu/_C.so.manifest.json`, `runtime/qualified/m04/manifest.json`.

The review independently checked the sizes and SHA256 hashes of all 2344 manifest-listed files, without mismatches. This establishes archive integrity, not GPU correctness. No new B70 experiment was executed during this review. Historical timings remain historical and all proposed gains remain unmeasured.

**Important rebuild trap:** `runtime/qualified/SOURCE_GUARD_COVERAGE.json` shows that the installed `qwen3_5_mtp.py` differs from the workspace version, and `b70_mtp_policy.py` is not present in that workspace. Preserve the installed native-EXL3 MTP policy. Do not reconstruct a candidate from a generic workspace and accidentally restore GPTQ/BF16 overrides.

### Existing negative evidence: do not repeat without a changed hypothesis

`E/docs/PROGRESS.md:192–214,247–265` records already tuned SmallM split-K, rejected vector M3/M4, rejected MB4, rejected whole-linear fusion, and rejected async scheduling / `FULL_AND_PIECEWISE` trials. In particular:

- MB4 gave an improvement within noise; it was not established as a speedup over MB8.
- Whole-linear fusion was slower under graphs, including M4 approximately 31.4 → 45.9 ms in one historical full-linear budget. The serialized split-K tail mattered.
- Enabling more piecewise graphs hit `UR OUT_OF_RESOURCES`; async scheduling did not establish a serving gain in the saved C8/C16 trial.

These are historical configurations, not proof that future redesigns cannot work. A retry must identify what changed and why the earlier result should no longer apply.

## 2. Compatibility and update matrix

This is a planning matrix, not a claim that the proposed combinations are qualified.

| Candidate | vLLM / Torch requirements | SYCL, EXL3 and M04 impact | Graph / memory obligations | Decision |
|---|---|---|---|---|
| oneDNN 3.13.4 | Retain vLLM 0.31 and Torch 2.14.0 initially | Build candidate custom oneDNN with the pinned toolchain; rebuild EXL3 against matching headers and library. No M04 source change needed | Preserve guarded prefill exclusion from captured decode, exact-K/Q policy and bounded completion-aware cache | Early maintenance/correctness trial |
| GDN norm from #608 | Minimal adapter for actual v0.31 rank-3 XPU route; #58118 alone is insufficient | Small FP16 SYCL extension; same queue and ABI. EXL3 weights and M04 unchanged | Fake/meta registration, no compile-time exclusion, same output shape/strides; no new global scratch | First provisional decode kernel |
| QKV from #627 | Minimal model dispatch or narrow matcher; do not import #60539 wholesale | Small SYCL extension; preserve EXL3 QKV producer and V view | Existing fused Triton is control; actual C1/C4 rows and graph buckets; do not expand prefill liveness | Second provisional decode kernel |
| #630 M04 port | No blanket vLLM change | Rebuild M04 from its pinned headers plus a reviewed minimal patch | Original masks, packed GQA, FP8 scales, graph scratch and q2–5 only | Profile-gated |
| M04 native-layout epilogue | Existing dispatcher unchanged | New M04 output-layout specialization; separate from #630 | Correct alias schema and stores; no persistent scratch growth | Profile-gated custom experiment |
| EXL3 boundary fusion | Explicit new op/route, original linear fallback | Rebuild EXL3; retain quantization layout and recipe | Preserve FP16 semantic boundaries, split-K parallelism, graph lifetimes and KV pool | Only after component evidence |
| Torch 2.14.1 | Resolve vLLM/kernel-package pins; do not force-install incompatible wheels | Rebuild all ABI-sensitive extensions, including EXL3 and M04; rebuild/replace other affected XPU extensions compatibly | Recheck compilation, graphs, memory accounting and precise version guards | Later, no documented XPU speed rationale |
| oneDNN 3.14-rc | Concrete required fix absent in stable first | Separate native build and source audit; no incidental SYCL upgrade | Experimental runtime/profiling changes must not leak into production policy | Hold |
| Torch 2.15 nightly | Exact relevant commit and compatible vLLM/XPU dependency set | Full affected ABI rebuild in a separate image | Higher graph/compiler/cache regression surface | Hold |
| Level Zero loader 1.34 | Distinct from the GPU compute runtime | Keep actual GPU driver/IGC pinned for a loader-only trial | Resource/event/graph-lifetime reproduction required | Hold without a relevant issue |
| Compute runtime / IGC / oneAPI / host kernel | Current stack is not shown to be the bottleneck | No broad host rolling update; use matched supported packages | Host rollback, Xe/GuC or compilation-fix justification required | Hold |

As checked on 2026-10-10, vLLM 0.31.0 remains the latest listed stable release; Torch 2.14.1 and oneDNN 3.13.4 are available. Compute runtime 26.35.39758.10 and IGC 2.41.5 already match the recent official release set. These release facts do not establish performance benefits. [U1–U5, U11–U12]

## 3. P0 — bounded route, cycle and memory audit

**Hypothesis:** the useful next patch can be selected by measuring the actual critical path, without another full baseline qualification.

**Changed variable:** diagnostics only. Keep inference algorithms and settings unchanged. Profiling runs are not the performance result.

### Implementation

Create `R/benchmarks/experiments/b70-stack-20261010/` with an experiment manifest, explicit baseline digest, candidate IDs, patch receipts and a small controller using the existing GPU lock/restore discipline. Proposed filenames below are implementation tasks, not scripts already present.

Implement `inventory.py`, `profile_decode.py`, `analyze_profile.py`, and `gates.json`. Reuse `R/scripts/collect-runtime-manifest.py`, the current-profile benchmark, prefix-state checks and existing numerical harnesses instead of duplicating them.

Use the qualified runner actually imported by the image. `R/scripts/b70_cycle_replay.py` is a **V2-runner** diagnostic: reuse its state-restoration principles, not its imports or runner selection blindly. Start from the installed-runner capture/replay adaptation already used by `R/benchmarks/experiments/vllm-031-upgrade/run-replay.py` and `native_capture.py`.

Record these separate spans and actual `(M,N,K)` or token/head shapes:

1. Target graph: EXL3 linears, Hadamard input/output transforms, GDN core/state operations, GDN norm, full-attention QKV postprocessing, M04 forward, M04 reduction, packing/copies, other normalization/activation work.
2. Outside the target graph: target LM head, each of the three autoregressive MTP steps and its LM head, proposal/sampling/rejection, state commit, synchronization and transfers.
3. Host: scheduling/metadata preparation, graph selection/submission, output processing and device-idle intervals. Distinguish enqueue time from GPU execution and waiting time.
4. Prefill only: EXL3 W8A8 route, oneDNN SDPA implementation, exact-K gathering/dequantization and bounded partition-cache behavior.

Require a **route manifest**, not only a Python call count: an FX/compiled-graph node or captured kernel name proving that the new op is actually executed. Capture-time Python counters do not count graph replays. Avoid GPU counters or synchronizing instrumentation in the final throughput run.

A bounded initial panel is two representative saved decode fixtures: C1 at 64K and C4 at the existing medium-context/prose fixture. Add one short C1 only when it separates launch-bound from long-attention behavior. Capture representative steady-state rounds, not the whole prefill, and run an uninstrumented control window to detect profiler distortion. Treat graph-replay timing as primary; eager timing only identifies components.

Use queue/device events or a verified XPU timeline. Do not add nested/overlapping kernel durations or interpret CPU waits as idle GPU time. If the chosen profiler cannot resolve graph child kernels correctly, report that limitation and use a representative whole-graph replay plus separate operator experiments; do not invent component shares.

Record allocated/reserved VRAM, graph-private pools, native extension scratch, oneDNN cache entries/evictions, KV blocks/bytes per group, admitted/running/waiting requests, preemptions and recomputation. `max_model_len=262144` alone is not a capacity result.

### Stop / go

Produce a ranked component budget and an estimated maximum end-to-end effect before implementation. For a non-overlapping fraction `f` sped up by `s`, use the *screening* bound `1 / ((1-f) + f/s)`. This is not a prediction for changing acceptance or overlapping work. An operation occupying 2% of the measured critical path cannot alone justify a promised 10% decode gain.

Provisional E02-before-E03 ordering may be reversed by this report. If both norms are negligible, move directly to a measured M04/EXL3 opportunity rather than implement them for completeness.

## 4. D64 — distinguish acceptance loss from higher cycle latency

**No serving implementation change in this diagnostic.** Reuse the existing archived quality and matched-state results first.

The saved `R/benchmarks/results/vllm-031-upgrade/performance-review-measurements.json` contains:

| 64K C1 metric | Historical v2 | Qualified v0.31 | New-image repeat |
|---|---:|---:|---:|
| Native decode, median tok/s | 52.6734 | 48.2944 | 48.3097 |
| Aggregate acceptance counter | 60.7034% | 56.0500% | 56.0500% |
| Same-window round-equivalent median | 54.5638 ms | 54.2386 ms | 54.2628 ms |

The new-image outputs repeat across corresponding fixtures; old and new outputs differ. These are three frozen tasks/repeats, not three interchangeable random draws. The historical baseline was not rerun. The round-equivalent is not a device-only physical iteration profile. Independent medians and aggregate acceptance must not be multiplied into an exact causal explanation.

Implement at most a small set of teacher-forced, identical token prefixes around the first divergence and one later long-context position. Compare the target logits, each MTP proposal, actual sampler transformations and, when applicable, p/q rejection decisions. Restore KV pages, GDN/convolution state, block tables, request-slot mappings, draft-to-target inputs and RNG state before repeated replay. A frozen output copy is not a substitute for checking the real computed result before the copy.

For within-version kernel A/B comparisons use the same captured inputs. For cross-version comparisons reconstruct each version's state from the same token IDs using its own compatible runner; do not load process-specific graph objects or assume identical state-buffer schemas. An old-image diagnostic run is allowed only when a named missing reference cannot be resolved from the saved evidence, and only for that narrow question.

Separate:

- **Distribution/trajectory:** target numerical changes, MTP proposal changes, tokenizer/chat/sampling configuration or RNG differences.
- **Iteration efficiency:** actual emitted/committed tokens per round and wall time for those same rounds.
- **State bug:** unexplained same-prefix divergence, wrong commit/rejection state, stale slot or cache reuse.

Use `sum(committed output tokens) / sum(serialized decode-window time)` alongside the existing client and engine definitions; exclude EOS/finish/padding artifacts consistently. Preserve the configured sampling distribution. Do not improve acceptance by changing temperature, top-p, draft depth, head truncation or workloads.

**Go:** fix one localized regression only after a falsifiable reference test identifies it. **Stop:** if the saved and bounded new comparisons support harmless trajectory differences within accepted quality limits, record that and return to latency optimization. Do not turn this into an unbounded cross-version bisect.

## 5. E01 — oneDNN stable maintenance and SDPA correctness

**Hypothesis:** the stable 3.13.4 line removes a relevant SDPA race and preserves the serving contract. Any prefill improvement is an empirical secondary outcome; direct decode acceleration is not assumed.

**Single logical variable:** the custom oneDNN version, with the EXL3 rebuild required for that dependency. Keep Torch, SYCL, M04, model, recipe and dispatch policy fixed.

### Review finding and uncertainty

The exact public upstream source at reported Git `0e2a5bfeef1bfbffc3137464606540233086ce9b` still initializes `S_max_slm` **after** the early Q barrier-arrive sequence in `src/gpu/intel/sdpa/micro.cl`. It does not contain the ordering change in fix `b38b47c9af53e2ad744241613728fa6cc545ec06`. The fix moves this initialization before the barrier-arrive sequence. oneDNN 3.13.3 documents the SDPA correctness repair; 3.13.4 is the later stable patch release. [U1–U3]

This resolves the *public source* question, not the complete production-binary question. The archive records the library hash but does not include the oneDNN source tree or binary. A locally patched/dirty build could retain the reported Git hash. Nor does source inspection prove that the affected micro-SDPA specialization and compile-time barrier branch is selected for each B70 shape.

### Implementation

1. Before rebuilding, identify the loaded custom library through the process mappings and `dnnl_version()`, not only `ldd` on a different file. Match runtime hash `ee8fc42adcf851ea0f8013ebda24f07629f49ca6ed815835a16da45ad718ca20` to its build receipt and any dirty patches. Separate Torch's internal oneDNN from `/opt/b70/libdnnl.so.3`.
2. Recover source and CMake/build flags. Check the exact initialization/barrier ordering, not merely `git merge-base --is-ancestor`, because equivalent cherry-picks can have another commit ID.
3. In a short **diagnostic**, use supported oneDNN implementation logging or retained compiled-kernel evidence to identify the path for actual FP16 prefill shapes. Do not enable verbose profiling on a queue that lacks the required capabilities; use ordinary implementation logging first. Keep diagnostic logging out of serving timings.
4. Build v3.13.4 in an isolated prefix with the existing oneAPI 2026.1.1 compiler/runtime and original GPU/Graph build features. Resolve and record the tag's complete commit and source checksum. Rebuild `E/csrc/exl3_ops.sycl` via `E/scripts/build_ext.sh` against matching candidate headers and library. Retain build flags unless required for compatibility and document any such changes.
5. Keep `E/exl3xpu/attention_dispatch.py:57–97,132–176`, `E/exl3xpu/fp8kv_prefill.py`, `E/csrc/exl3_ops.sycl:398–573` and `E/csrc/bounded_partition_cache.h` behavior unchanged. Preserve cache capacity 64, completion-aware reuse/eviction, queue/context binding, exact K, validated Q padding and one compiled SDPA partition. Do not permit a large unfused score-matrix partition as a silent fallback.
6. Verify headers, loaded runtime version/path/hash, RPATH, registrations, `ldd`, and actual queue interoperability. Prevent symbol/path collisions with Torch's internal libraries. Rebuild the candidate receipt and guard hashes; never overwrite the production artifact in place.

If stable 3.13.4 introduces unrelated incompatibility or a clear regression and the old source is proven unpatched, create **E01-minimal** from the same old source plus only `b38b47c...` (or its reviewed equivalent), preserving existing local patches. This is a separate fallback candidate, not a second mandatory version matrix.

### Tests and decision

Run short independent attention references, exact-K/chunk/page boundaries, real FP8-KV scales, cold and warm prefill, mixed prefill/decode, cache pressure above 64 distinct partition keys, repeated eviction with in-flight work, and the existing near-limit capacity gate. Test INT8 prefill matmul as well: EXL3 links more than the SDPA entry point to oneDNN.

Only then run the unchanged small serving panel, including C1/C4 decode non-regression. A documented correctness improvement may justify a maintenance promotion without a speed gain, provided capacity and performance remain within the maintenance envelope. Label it accordingly. Stop for an unexpected implementation fallback, a missing local vendor feature, unbounded workspace/cache growth, numerical failure, or context loss.

The documented 3.13.3 SDPA *performance* repair is for `f32`; internal FP32 scores do not make this FP16 activation path automatically equivalent to that workload. Do not promise a benefit from that release-note item. [U2]

## 6. E02 — graph-active FP16 GDN gated RMSNorm

**Hypothesis:** a specialized row-wise SYCL implementation lowers the actual GDN output-norm cost under MTP3 graphs.

**Single logical variable:** replace only the existing gated norm implementation for an explicitly eligible XPU decode shape. Do not remove zero initialization, fuse output projection, change GDN state kernels or add FP8 quantization in this candidate.

### Exact source basis

- Kernel: `P/vllm-project-vllm-xpu-kernels-608.diff`, gated-norm hunk; head `19cbfde65788f9c4643d394cd619f22bc22ee8c5`.
- Reference dispatch: `P/vllm-project-vllm-58118.diff`; head `14a652d46f6d4c03872486b9b362169b522618b6`.
- Actual route: `V/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:490–496,845–857,972–1010` and `V/model_executor/layers/layernorm.py:216–352`.

The supplied kernel is generic over the number of 128-element rows. Its Python guard is not: it accepts only 2-D `(16,128)` / `(24,128)` and excludes compilation. The actual XPU GDN path supplies `(T,48,128)` to `self.norm`. Adding `(48,128)` alone therefore does **not** implement support for this target. A Python function that never runs in the compiled graph cannot deliver a decode benefit. [Archive P files; U6–U7]

### Implementation

Extract only the gated-norm kernel, declaration, registration and tests into a small candidate extension, preferably a non-colliding `b70_stack` namespace. Use the same Torch XPU stream/queue and DeviceGuard convention as the qualified native extensions. Retain all-device/all-dtype checks, contiguity, width 128, positive finite epsilon and no untracked output alias.

At the actual `_output_projection` boundary, use a guarded adapter that verifies `FP16`, `(T,48,128)`, contiguous x/z, a contiguous FP16 `(128,)` weight, `group_size=None`, `bias=None`, `norm_before_gate=True`, and SiLU/swish activation. Flatten as a **view** to `(T*48,128)` and reshape the output back. Fall back for unsupported shapes; do not insert implicit `.contiguous()` copies into the fast path.

The relevant counts are:

| Work | T | Norm rows |
|---|---:|---:|
| Single-token work | 1 | 48 |
| Full MTP3 target verification, C1 | 4 | 192 |
| Full MTP3 target verification, C4 | 16 | 768 |
| Largest current 64-token graph bucket | 64 | 3072 |

These are shape cases, not an assertion that every scheduled round is full q4. The trace must include actual shortened/padded cases and separate target/draft routes.

The decode bound applies to **T**, not the flattened row count. Register a correct fake/meta implementation and mutation/alias schema, and ensure the adapter is reachable during compilation and graph capture. Use either the explicit XPU GDN call-site adapter or a verified narrow matcher. Do not retain `not torch.compiler.is_compiling()` as the only route. Preserve native prefill behavior and output lifetimes; no broad global norm replacement.

### Numerical contract

For each 128-element row, compute in FP32:

```
inv = rsqrt(mean(x_fp32 * x_fp32) + epsilon)
y_fp32 = (x_fp32 * inv * weight_fp32) * silu(z_fp32)
y = cast_fp16(y_fp32)
```

There is **no `1 + weight` here** and no extra FP16 cast after normalization. That contract differs from the QKV norm/RoPE rounding boundary in E03. Use the existing native/reference tolerances and real captured tensors, including tiny variance, large finite inputs, large positive/negative gates, zeros and mixed signs. Require a same-input reference, not merely finite output.

### Test / stop / go

Test zero rows, T1/4/16/64, actual graph sizes and padding, stride rejection, compilation and repeated graph replay with changed input buffers. Exercise independent request slots, rejection/state restoration and mixed-prefill fallback. Inspect the graph/kernel names to confirm use.

Compare the standalone captured norm and the enclosing captured GDN/target graph before serving. Stop if the wrapper falls back, adds graph breaks/copies, increases prefill memory, or cannot produce a material whole-cycle benefit. The author’s eager host-dispatch gains are not transferable speed estimates for this graph-heavy stack.

## 7. E03 — replace the existing fused Triton QKV postprocessing

**Hypothesis:** the specialized SYCL implementation is faster than the already fused Triton kernel on this B70, while preserving model arithmetic and buffers.

**Single logical variable:** QKV postprocessing implementation. The EXL3 QKV projection, attention, KV write and MTP policy remain unchanged.

### Source and adaptation

Kernel #627 head: `320a7a58d8a70f435961dec1992cc9b83f64ef42`. Companion vLLM #60539 reviewed head: `8f3e461beebc37d665a2cc2ba805adea1617dc34`.

Actual call site: `V/model_executor/models/qwen3_next.py:364–425`. Existing implementation: `V/model_executor/layers/fused_qk_norm_rope.py:17–280`.

The deployed model already calls `fused_qk_rmsnorm_rope_gate` on eligible XPU shapes. The proposal is not “three separate launches → one.” The companion bundle's latest archived revision includes a matcher for that existing fused pattern, but imports many unrelated operations. Take only the required wrapper/fake/dispatch logic or implement a smaller explicit call-site adapter. [U8–U9]

The actual full-attention geometry is **24 Q heads, 4 KV heads, dimension 256: GQA 6**, not the DFlash draft geometry. Preserve gated row ordering, Q/K norm `weight_offset=1.0`, the original epsilon, actual partial rotary dimension and FP16 cached cos/sin. Keep V as the original valid view; do not reconstruct the QKV tensor with a concatenation merely to call the new kernel.

The upstream narrow companion matcher caps token rows at 8; C4 q4 normally has T16. A kernel unit test up to T64 does not expand its dispatch. Implement a bounded, graph-safe eligibility policy covering the actual T4/T16 decode graphs and existing capture sizes when qualified. Do not blindly remove every bound or fuse large prefill ranges: increased live outputs can reduce the KV pool.

The C++ op requires int64 positions and particular strides and rotary divisibility. Existing Triton support is not identical. Fall back for unsupported inputs rather than converting/copying on every step. Preserve the model's existing handling of 2-D positions and text-versus-multimodal dispatch. Add same-XPU-device checks for **all** passed tensors and output buffers; the reviewed #627 entry has extensive shape/dtype checks but does not comprehensively verify every device. Validate positive head counts and buffer-size arithmetic too.

### Numerical and semantic tests

Use an independent reference that preserves the current order: full-head FP32 RMS normalization, weight offset, **round norm output to FP16 before RoPE**, then rotation with the same FP16 cos/sin cache. Check the unrotated tail separately. Gating is copied raw, not activated here. V must remain unchanged.

Test actual gated FP16 Hq24/Hkv4/D256 shapes at T1/4/16 and all relevant graph buckets; nontrivial row strides accepted by the op; rejected inner strides; ordinary RoPE, supported interleaved MRoPE and fallback configurations. Include positions around 65535/65536, 131071/131072 and 262143, plus actual page boundaries. Positions are indices, not FP16 values; never cast them through FP16. Test media-related positions through the existing image/video smoke harness.

Compare q/k/gate/V, then downstream attention and target/MTP distributions on fixed inputs. Unchanged greedy text alone is not sufficient; neither is a small hidden-state error without checking tight logit decisions.

### Stop / go

Require measured SYCL op usage in captured C1 **and C4**, positive full-graph benefit, unchanged prefill fallback, no KV-capacity reduction, and the serving decision gates. If the old Triton kernel is already equally fast under capture, stop. Do not add unrelated FP8 GEMV, MoE router, all-reduce, zero-init removal or checkpoint-role fusion from #60539/#622 to manufacture a win.

## 8. E04 — M04-specific attention opportunities

### E04a: port head-dimension reduction tiling only

**Hypothesis:** M04's split-K reduction occupies enough of long-context decode to benefit from additional head-dimension tiling.

**Single logical variable:** reduction grid/tiling; retain the current attention mainloop, split-count policy, packing, dtype, masks and serving dispatcher.

Upstream idea: kernel #630 head `bbe92f3a2829da16b5dd3798cf2f3c0a5a61655e`. Current sources:

- `runtime/qualified/m04/csrc/shared_kv_verification.cpp:1–155`
- `runtime/qualified/m04/patches/shared-kv-verification.patch`
- `kernel-base/csrc/xpu/attn/xe_2/kernel/paged_decode_kernel.hpp`, particularly `ReduceSplitK` at 532 onward and reduction/store at 792–851
- `E/exl3xpu/shared_kv_verify.py:21–120`
- `E/exl3xpu/attention_dispatch.py:90–97`

An ordinary upgrade of `vllm-xpu-kernels` does not replace the separately compiled M04 `.so`. Port the idea onto the exact pinned header tree and rebuild M04 using the existing C++20 build wrapper. Do not install the full upstream bundle and assume the primary path changed. Generic fallback attention can be affected by a package update, but that is a different candidate; the outer M04 dispatcher currently admits C1–C4 only. [U10]

The current original M04 accepts q2–5. For normal MTP3, q4 is the primary verification case; q2/q3 and fallback behavior still need coverage. Keep q5 compatibility if retaining the original operator contract, without launching another MTP4 serving campaign.

Preserve the original packed mapping `q.view(B,R,4,6,256).permute(0,2,1,3,4)`, the per-verification-row lower-right causal history, real device sequence lengths, scalar FP8 K/V scales and scratch shapes. The current policy uses `{2:32,3:8,4:16,5:16}` splits and tile8; do not change this simultaneously with the reduction. In particular, do not casually import the DFlash-wide adaptive-split implementation as the qualified baseline.

The reduction combines **unnormalized** partial outputs. Preserve its max/exp rescaling convention (including log base), exclude empty splits, and divide once by the combined exp sum. Do not accidentally normalize partial results twice. Every head-dimension tile must get valid statistics and write disjoint output elements.

Run original M04 numerical, fallback and graph tests against native attention and an independent short-history FP32 oracle with the same quantized KV values. Add adversarial unequal request lengths, inactive splits, page edges, near-262K positions and changing device lengths across replay. Measure mainloop, reduction and pack/copy separately, then the whole cycle. Stop if reduction is negligible or tiling causes extra overhead, scratch growth or a short-context regression.

### E04b: direct native-layout output, a separate custom-kernel experiment

**Hypothesis:** eliminating the packed-output materialization and follow-up copies saves useful graph time without changing attention math.

Current `shared_kv_verify.py:97–119` allocates packed `y`. C4 uses a direct strided copy from that packed output; other cases create the unpacked contiguous layout and then copy into `out`. The historical C1-versus-C4 copy result is already encoded in the source, so merely applying the C4 copy everywhere is not a new optimization.

Instead, after profiling proves a material cost, add a **new M04 reduction-output-layout specialization** that stores directly into the supplied native `[B*R,24,256]` output. Retain old behavior as fallback and keep the reduction algorithm unchanged in this experiment.

For packed head index `h = g*(R*6) + r*6 + s`, where `g in [0,4)`, `r in [0,R)` and `s in [0,6)`, write dimension `d` to:

```
native_offset = ((b*R + r)*24 + g*6 + s)*256 + d
```

Verify this mapping independently before touching device code. Keep packed scratch indexing unchanged. Initially support only the qualified multi-split path; a single-split path writes in the attention mainloop and requires its own output-layout treatment, so retain its original fallback until separately tested. Do not alias the output with partial scratch.

The stored values should be bitwise identical when only addressing changes. Test request boundaries, all R2–5, B1/B4, changing lengths, padding and graph replay. Benchmark the complete M04 call including old packing/copies. Any surviving input-pack optimization or fusion with the QKV producer is a later, separate dependency, not part of this first patch.

## 9. E05 — EXL3-specific custom kernels, only beyond prior tuning

**Hypothesis:** a measured producer/consumer boundary or a newly demonstrated bottleneck in the native EXL3 route has a larger potential benefit than generic FP8 kernels.

**Single logical variable:** exactly one selected EXL3 boundary or shape specialization, identified in the P0 report. Do not combine weight-format changes with a kernel optimization.

The dominant decode linears use the native EXL3 operation, not an ordinary FP8 matrix. See `E/exl3xpu/ops.py:63–138` and `E/csrc/exl3_ops.sycl:217–330,701–807`. The separate W8A8 prefill and SmallM routes must remain separate. The native file already records `g_fused=false` because the earlier fused split-K tail was slower under graphs.

### Preferred first design if the measured boundary supports it

Explore **GDN gated norm + EXL3 input Hadamard preparation**, not norm + an unrelated FP8 quantization kernel. Source boundaries are `_output_projection` in the actual GDN model and EXL3's HadIn/SmallM path. The potential saving is the normalized intermediate write/read and one launch, not a reduction in the number of EXL3 weights decoded.

This is an original design hypothesis, not a validated patch. Before implementing:

- Map the model's 48 groups of 128 normalization elements to the real EXL3 input dimensions, Hadamard group boundaries and `suh` layout. Inspect actual recipe tensors; do not assume the groups line up.
- Keep normalization, weight and SiLU arithmetic in FP32 until the exact existing FP16 normalized-input boundary. Even inside a fused kernel, explicitly reproduce that FP16 rounding before the Hadamard transform.
- Compute each needed norm reduction once at the input-preparation stage. Do not repeat a row's normalization in every N tile or every split-K worker.
- Preserve the existing padded-row initialization, FP16 activation policy, transform signs/scales, group descriptors, codebooks and output transform.
- Do not merge the parallel GEMM and last-arriver tail into the already rejected full-linear design.

A less ambitious alternative is one measured output-Hadamard/activation epilogue or residual/norm boundary. Choose **one**, with actual graph time and tensor-liveness evidence. Sharing a Hadamard transform across projections is legal only if the corresponding transforms, groups and sign tensors are identical; common input activations are not enough.

### Boundaries and stop rules

Use `E/bench/linear_budget.py` and the existing real-weight kernel gates as a starting point. Focus on actual M1/M4/M16 and the measured dominant layer types. The historical split-K plateau, MB4 negative result and vector-M4 loss are not reasons to sweep those again. A compiler-induced code-generation change may justify a precisely stated retest, but then the compiler is the separate candidate variable.

Reject global dequantized-weight caches, lower-bpw re-quantization, changed LM-head support and BF16 intermediates. Do not prune the full vocabulary or replace sampling with argmax to make a head benchmark faster. Native target and autoregressive MTP heads must still implement the configured sampling semantics.

Compare against an independent decoded-weight/reference computation with identical transforms, as well as the old kernel. Verify mixed quantization groups, layer roles, padding, repeated graph state and the exact recipe hashes. Stop if materialization savings are below measurement noise, register pressure erases the saving, or graph memory reduces usable KV capacity.

## 10. E06 — scheduling and graph work only after a real idle-time diagnosis

**Hypothesis:** one specific avoidable host/metadata operation delays device submission or blocks concurrent request progress.

**Single logical variable:** one identified host operation or one metadata preparation strategy. Preserve MTP3, graph mode, capture sizes, token budget, admission and cache policy.

Inspect the installed runner and worker identified by the route manifest and source guards, `E/exl3xpu/attention_dispatch.py` / metadata helper, and the current benchmark's engine/client timestamps. Prefer eliminating an unnecessary D2H scalar read or repeated immutable metadata construction over changing the whole scheduler.

All device-derived lengths and acceptance-dependent state must remain valid across replay. Do not reuse stale host lengths, remove required synchronization, or cache metadata by graph size alone when request identity/block tables differ. New host caches must have bounded lifetime and explicit invalidation.

Historical async scheduling and larger graph-mode trials did not establish a win. Retest only with a concrete fix for the observed earlier failure or a new trace showing a different limiting path. Do not call an eager GPU-sum versus wall-time difference an “async opportunity” without a graph-mode timeline.

A hand-built single MTP-step graph or a graph with sampler dependencies could be investigated later only if uncaptured launch gaps dominate and captured inputs/outputs, RNG, state commits and graph resources can be kept correct. Do not capture many layer-wise subgraphs or all acceptance outcomes by default.

For mixed prefill/decode contention, first report decode TPOT/ITL, TTFT and queue occupancy separately. A change in chunk size or admission is a distinct policy experiment, not a free kernel gain; it must preserve admitted concurrency and capacity. The current 4096-token budget and graph sizes already have historical context, so avoid a broad parameter sweep.

**Go:** smaller genuine device-idle gaps plus better C1/C4 serving without queue starvation. **Stop:** timing only moves between CPU wait categories, GPU work is unchanged, or C4 waiting/aggregate throughput regresses.

## 11. E07 — Torch 2.14.1, later and separately

**Hypothesis:** a clean patch-level maintenance upgrade remains compatible; performance is measured, not assumed. The public release notes describe CUDA and MPS fixes, not an XPU decode optimization. [U4]

**Single logical variable:** Torch 2.14.1 and its strictly required matching build artifacts. Do not simultaneously update oneDNN, Triton, SYCL, the driver or unrelated vLLM code to repair the dependency tree invisibly.

Resolve Python 3.12 XPU wheels and the complete package constraints before building. A conflicting exact vLLM/XPU-kernel pin must be resolved through an explicit compatible source build and recorded package metadata, not `--no-deps`, forced installation or ignored `pip check` output.

Rebuild EXL3, M04 and every other ABI-sensitive extension against the candidate Torch headers/libraries. A matching `_GLIBCXX_USE_CXX11_ABI` value alone does not establish C++ binary compatibility. Record `torch.__version__`, Torch Git/build configuration, compiler, linked libraries and native artifact hashes.

Update candidate-specific assertions in at least:

- `R/scripts/run-server-exl3.py` runtime profile allowlist;
- `E/exl3xpu/attention_dispatch.py:140–145`;
- `R/benchmarks/experiments/vllm-031-upgrade/build-m04.sh:10` and its candidate copy;
- builder package/version checks and the native artifact/recipe receipts.

Search the complete candidate tree for exact Torch pins before launch. Preserve the installed `qwen3_5_mtp.py` and `b70_mtp_policy.py`, rather than replacing them with workspace files. Add new affected source files to the guard manifest without weakening guards for production.

Run ABI/import, changed-stack numerics, graph replay, matched-state quality, original capacity and a targeted serving panel. Broaden to the full frozen panel only if considering promotion. Stop if the upgrade requires a much larger unplanned dependency migration or yields no maintenance rationale with measurable regressions.

oneDNN 3.14-rc or Torch 2.15 nightly becomes reasonable only with an exact relevant fix unavailable in the stable combination, a reproducer and a minimal dependency plan. Resolve an immutable nightly/wheel hash at execution time; “latest nightly” is not a reproducible experiment.

## 12. Common build and source-guard protocol

Every candidate is a separate image/worktree and a separate patch. Keep the production image and policy immutable. Use the original fixture/model hashes. Do not upgrade host packages during container experiments.

For native work, adapt the qualified build wrappers rather than invent a looser build:

- `E/scripts/build_ext.sh` builds EXL3 with the pinned compiler and creates a native manifest; require oneDNN explicitly for E01 instead of accepting a headers-missing fallback with oneDNN disabled.
- `R/benchmarks/experiments/vllm-031-upgrade/build-m04.sh` applies the original M04 patch with `--fuzz=0`, uses C++20, the required SPIR-V extensions and Torch XPU linking, checks registration, and writes a manifest.
- A small independent `b70_stack` extension for E02/E03 should use the same toolchain/ABI, explicit op schemas and fake/meta registrations. Avoid replacing the entire vllm-xpu-kernels package merely to add one operation.

Record exact compiler argv, CXX11 ABI, source and patch hashes, header revisions, wheel hashes, linker/RPATH details, library hashes, registrations and CPU import checks. Compile success must remain labeled **GPU untested** until GPU tests pass. Do not loosen source guards or silently retry a failed fast path through a different implementation.

Add changed `layernorm.py`, GDN/model dispatch, `fused_qk_norm_rope.py` and new wrapper/kernel files to candidate integrity coverage as applicable. Unchanged production guards remain unchanged. Fail startup if the requested op is missing, source hashes do not match or a candidate library is not the one actually mapped in the process.

The rollback procedure is to stop the candidate through the existing controlled runner and restart the pinned qualified image/profile. No copying candidate `.so` files into the production image, no shared mutable build directory, and no runtime `LD_LIBRARY_PATH` workaround whose effect is unrecorded.

## 13. Progressive correctness, capacity and performance gates

### G0 — provenance, CPU, import, ABI

Verify the input manifest, patch applicability, candidate dependency resolution, source guards, build receipts, imports, schema/fake support, actual mapped libraries and XPU registrations. Exercise invalid-input guards without dereferencing wrong-device pointers. Stop before GPU work on a failure.

### G1 — independent operator references

Use original tolerances and the existing reference framework. Never widen thresholds after seeing a candidate fail. Test random and real captured tensors; low-variance and large-value numerical cases; all relevant dtypes/layout rejection; padding and bounds; exact untouched tensors and aliases. Distinguish mathematical/reference comparisons from comparing two implementations that could share a bug.

For attention, first compare with a dense reference using the same quantized K/V; separately assess quantization versus higher-precision K/V. For EXL3, reconstruct reference weights/transforms without changing the model recipe. For addressing-only changes, demand exact store equivalence where appropriate.

### G2 — model and sampling/state

Check actual target and MTP outputs on fixed prefixes, full-vocabulary distributions at selected positions, relevant top decisions and the configured proposal/rejection path. Retain original saved quality references. Check rejection suffixes, repeated slots, prefix reuse, abort/restart and context-end state behavior. Finite activations or one matching greedy completion is not complete correctness.

Do not require bit-identical free-running text across all numerically valid implementations as a substitute for the established quality gates. Conversely, do not explain away a systematic same-input failure as harmless sampling noise.

### G3 — captured graph and live buffer behavior

Run actual q4 C1/C4 forms and the existing graph buckets, plus shortened/padded/mixed cases and native fallbacks. Change device input buffers and sequence lengths between replays; check no stale addresses or state. Confirm the candidate route survives compilation and capture. Keep a separate uncaptured diagnostic; it is not the performance winner criterion.

### G4 — memory and usable capacity

Keep the configured limit at 262144 total tokens and **exercise** the qualified boundary with tokenized input plus generated output, not a startup assertion or an oversized prompt that never decodes. Reuse the existing exact-boundary harness. Record per-group KV blocks/bytes, graph pools, partition caches, temporary workspaces and actual high-water marks before and after.

Retain admitted sequences 16 and the existing C1/C4 capacity/queue workload. This does not mean 16 simultaneous 262K sequences fit: do not make that unsupported claim. Compare the same operational workload and scheduler reservation policy. Any lower usable KV pool needs explicit accounting and is not accepted as a default tradeoff.

No new OOM, driver reset, graph resource exhaustion, unbounded cache growth, extra preemptions or recomputation on cases that had none. Where historical workloads have baseline preemptions, compare their count and recompute work instead of claiming the baseline never preempts.

### G5 — targeted serving screen

Reuse the frozen prompts, sampling, seeds, token limits and cold/warm definitions. Minimum coverage is short C1, 32K/64K C1 (including the known 64K fixture), long C1, and representative source/code and prose C4 cold/warm. Add near-limit serving as G4 requires and a limited 16-admission queue smoke. Use the already defined scenarios; do not invent a new easy panel to favor a kernel.

Do **not** run the full historical baseline campaign first. Candidate-versus-qualified-image paired control windows in this small panel are justified to measure the new patch under the same conditions. Reuse unchanged numerical references. Expansion to the full frozen panel belongs to the final combined-candidate decision, not every failed patch.

Use three independent paired blocks for screening, with order balanced or randomized and prefix state reset/verified inside each arm. Repeated graph iterations in one process are not independent serving repetitions. Preserve per-fixture effects and output/acceptance differences. For a credible survivor, confirm in additional independent blocks (typically at least five total; extend when uncertainty requires it). An inconclusive confidence interval is not a win.

Keep cache state verified via `R/scripts/check-prefix-state.py`; first/second execution does not define cold/warm. Keep the GPU exclusive and record clocks, throttling, power and temperature only to validate comparability, not to rerun the power matrix. Profiler/capture logging must be off in throughput measurements.

### Required metrics

Retain raw engine and client data and definitions for TTFT; native decode rate; per-request TPOT/ITL where available; aggregate C4 client throughput; E2E median and tail latency; actual emitted tokens per round; acceptance/proposal counters; full-round wall time; peak VRAM; KV pool; Running/Waiting; preemptions; recomputation; route/fallback counts; and output hashes/quality results. Do not replace a losing client metric with a winning native metric.

### Pre-register the decision thresholds

The following are proposed engineering rules, not already achieved results:

- **Performance promotion:** at least a 3% weighted decode gain over the relevant qualified control, with paired uncertainty supporting a positive effect, and no credible >3% regression in a critical C1 or aggregate-C4 decode cell.
- **Latency/prefill guardrail:** no credible >5% TTFT, prefill-throughput or E2E-tail regression in protected cases. Investigate smaller systematic regressions; thresholds are not permission to hide them.
- **Capacity/correctness:** mandatory; not compensated by a higher throughput average.
- **Maintenance promotion:** an identified relevant correctness/maintenance benefit may pass without 3% speedup, but still requires non-regression/capacity checks and must not be reported as a performance win.
- **Microbenchmark survivor:** a repeatable operator improvement that has a plausible whole-cycle effect can continue even before meeting the 3% standalone serving threshold. A small, reliable, non-regressing gain may be retained as a building block, but it is not a production winner until evaluated in the combined candidate.

Freeze workload weights from the established user-relevant benchmark before seeing results. Report every protected cell separately, including prose C4. Use paired per-fixture ratios or log-ratios and uncertainty over independent blocks; do not pool thousands of tokens as independent samples or mask C4 with a C1-weighted headline.

## 14. Combining winners and stopping the campaign

Combine only individually understood, compatible survivors, one integration step at a time. Record whether the control now includes a promoted oneDNN maintenance fix. A combination may alter compilation, liveness or graph memory, so its speedup is not the sum of the individual percentages.

Run the changed-stack numerical/quality and capacity gates, then the complete frozen serving panel for the **final** candidate. Retain the original MTP3 image as immediate rollback. Produce a release decision with per-cell results and uncertainty, not an automatic promotion from a kernel benchmark.

Stop the campaign when the remaining measured component opportunities cannot plausibly produce useful gains without violating constraints. A defensible result that a newer library is neutral or a custom kernel loses is useful. Do not reopen DFlash, MTP4+, BF16 bridges, lower context, altered sampling, whole-host upgrades or lower-bpw weights as a substitute for a negative finding.

## 15. Required Codex deliverables and first commit

The first implementation commit should contain **only diagnostic/experiment infrastructure**: pinned manifests, route/cycle profiling, oneDNN source-to-binary audit, the bounded D64 comparison specification, and pre-registered gates. It must not alter production defaults.

For every later candidate, deliver a small patch, exact source/ABI build receipt, independent test results, graph-route evidence, memory ledger, raw serving results and a one-page decision with one of: `INACTIVE_ROUTE`, `CORRECTNESS_FAIL`, `CAPACITY_FAIL`, `NO_MATERIAL_GAIN`, `INCONCLUSIVE`, `MAINTENANCE_SURVIVOR`, or `PERFORMANCE_SURVIVOR`.

Suggested output structure:

```text
benchmarks/experiments/b70-stack-20261010/
  README.md
  baseline-manifest.json
  gates.json
  inventory.py
  profile_decode.py
  analyze_profile.py
  candidates/E01-oneDNN/ ... E07-torch/
  patches/

benchmarks/results/b70-stack-20261010/
  P0-route-and-cost-report.md
  D64-same-prefix-review.md
  <candidate>/build-receipt.json
  <candidate>/numerics.json
  <candidate>/graph-route.json
  <candidate>/memory-ledger.json
  <candidate>/raw-serving/
  <candidate>/decision.md
  FINAL_DECISION.md
```

The names are a proposed handoff interface, not a claim that these scripts already exist. Implement them by adapting the archived controllers, not by copying commands with stale image IDs, device nodes, mutable caches or runner imports.

## 16. Direct answers to the six review questions

**1. oneDNN 3.13.4 first?** First dependency candidate and early correctness/provenance check, yes; first promised decode optimization, no. The public source at the reported Git hash lacks the barrier-order fix. Dirty build contents and runtime specialization remain to be verified. Stable 3.13.4 first, minimal backport fallback if necessary, not a broad version sweep.

**2. SYCL QKV versus existing Triton?** A plausible replacement, not a missing fusion. Keep only #627 plus minimal graph-aware dispatch/fake support from #60539 or a smaller explicit adapter. Preserve FP16 rounding, MRoPE, V views and T16 C4 coverage. No gain is established before captured timings.

**3. GDN 48-head adaptation?** The arithmetic kernel can process generic row counts, but the proposed Python route misses both the actual rank-3 shape and compiled execution. Adapt the actual `(T,48,128)` route; verify 192/768 norm rows for C1/C4 q4, and prove graph reachability.

**4. Generic reduction #630?** It does not automatically change the separately built M04. Port only the relevant reduction idea if it has a material cost share. Generic fallback improvements are a separate possible effect. A direct native-output epilogue is an additional, separately testable original optimization.

**5. Torch 2.14.1 early?** Not for an evidenced decode gain: release notes provide no XPU performance fix. It is a later maintenance experiment with full affected ABI/source-guard handling. RC/nightly trials need an applicable exact fix and a reproducer.

**6. Better than further depth experiments?** A measured graph-active norm/postprocessing replacement, M04 reduction/output-layout optimization or a narrowly chosen EXL3 materialization boundary is more defensible under the fixed MTP3 contract. Host changes require actual avoidable device-idle time; known failed fusion/graph/scheduling options are not fresh experiments.

## 17. Source ledger

### User-provided archive evidence

Use the original archive and the companion `B70_MTP3_REVIEW_EVIDENCE.md` for exact excerpts/line numbers. Most implementation details above derive from the supplied source files and PR diffs, not an assumption that moving upstream `main` equals the deployed code.

Core inputs are `PRO_REVIEW_REQUEST.md`, `RESEARCH_AND_APPLICABILITY.md`, `CODE_MAP.md`, `SOURCE_PROVENANCE.json`, `R/config/production_image.json`, `runtime/qualified/CURRENT_LIBRARY_VERSIONS.json`, `runtime/qualified/SOURCE_GUARD_COVERAGE.json`, the qualified EXL3/M04 receipts and the source paths cited in each candidate.

### Primary upstream sources independently checked on 2026-10-10

URLs are recorded for reproducibility; resolve complete immutable release commits and file hashes before a build. The immutable PR heads above are the reviewed archive versions, not an instruction to pull a later head silently.

| ID | Source | Reproducible address |
|---|---|---|
| U1 | oneDNN stable 3.13.4 release | `https://github.com/uxlfoundation/oneDNN/releases/tag/v3.13.4` |
| U2 | oneDNN 3.13.3 fixes | `https://github.com/uxlfoundation/oneDNN/releases/tag/v3.13.3` |
| U3 | Exact old SDPA source and fix | `https://raw.githubusercontent.com/uxlfoundation/oneDNN/0e2a5bfeef1bfbffc3137464606540233086ce9b/src/gpu/intel/sdpa/micro.cl` ; `https://github.com/uxlfoundation/oneDNN/commit/b38b47c9af53e2ad744241613728fa6cc545ec06` |
| U4 | Torch 2.14.1 release | `https://github.com/pytorch/pytorch/releases/tag/v2.14.1` |
| U5 | vLLM stable release list | `https://github.com/vllm-project/vllm/releases` |
| U6 | Gated RMSNorm dispatch #58118 | `https://github.com/vllm-project/vllm/pull/58118` |
| U7 | Gated RMSNorm kernel / larger bundle #608 | `https://github.com/vllm-project/vllm-xpu-kernels/pull/608` |
| U8 | Standalone QKV kernel #627 | `https://github.com/vllm-project/vllm-xpu-kernels/pull/627` |
| U9 | Companion compile/dispatch bundle #60539 | `https://github.com/vllm-project/vllm/pull/60539` |
| U10 | Split-K reduction proposal #630 | `https://github.com/vllm-project/vllm-xpu-kernels/pull/630` |
| U11 | Intel compute runtime release list | `https://github.com/intel/compute-runtime/releases` |
| U12 | IGC 2.41.5 | `https://github.com/intel/intel-graphics-compiler/releases/tag/v2.41.5` |
| U13 | Level Zero loader 1.34 | `https://github.com/oneapi-src/level-zero/releases/tag/v1.34.0` |
| U14 | oneDNN 3.14 release candidate | `https://github.com/uxlfoundation/oneDNN/releases/tag/v3.14-rc` |
| U15 | Triton pointer canonicalization change | `https://github.com/intel/intel-xpu-backend-for-triton/pull/7515` |

Do not treat draft/open PRs as validated releases. Their existence supports an experiment, not a speedup claim.

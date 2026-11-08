# vLLM 0.33 qualification on the B70, 2026-11-08

**Qualified and deployed as the permanent 180 W production service.**
The approval is recorded in [decision.json](decision.json); the actual
[deployment and restart receipt](deployment/promotion.json) records both live starts.

The comparison holds the model revision, EXL3 quantization, native libraries,
Torch 2.14.0+xpu, Triton 3.8.0+xpu, Intel userspace and 180 W power cap fixed.
The candidate installs the official final vLLM 0.33.0 XPU wheel and necessary
Python dependency updates over the qualified 0.31 image. It ports the Python
integration to the new upstream interfaces; this is not a comparison of two
different Intel toolchains.

The serving profile retains FP16 activations, FP8 KV, MTP3 with the pruned draft
head, 262,144 total context, 16 sequences, 4,096 batched tokens, LBNHC, the 12 GiB
RAM prefix cache and completion reservation. The RAM overlay's effective
watermark is 0.05; the underlying model profile retains its 0.0 default.

## Method

Both versions replay the same frozen requests: 20 scenarios, 70 measured waves
and 124 requests of 1,024 output tokens each. The matrix covers single requests
from short prompts through 128K context, concurrency 2/3/4, cold/warm prefix
reuse and a 199,680-token input. Sampling is temperature 1.0, top-p 0.95,
top-k 20, fixed per-request seeds, thinking disabled and ignored EOS. Warmups
are separate. Full request dictionaries and prompt bytes must match.

The acceptance plan was fixed before candidate measurements. A median
regression above 5% in an individual cell, or above 2% in weighted throughput,
triggers investigation and matched follow-up measurements. These thresholds
are investigation criteria, not a statistical confidence bound or SLA.
Decode, prefill, complete request time, energy, draft acceptance, cache hits
and preemptions are examined together. Sampled output can change across
versions despite the same seed, affecting speculative acceptance and speed.

The separate correctness gates cover checkpoint loading, original-tolerance
attention and linear tests, actual hybrid state replay and graphs, API/tool/
structured-output/media behavior, maximum context, concurrency and admission,
abort recovery, natural CPU-cache restoration and restart. Tokenizer auditing
compares actual rendered token IDs for all 124 frozen requests.

## Completed serving comparison

[Full measurements](serving.json) validate all 70 waves and 124 requests on
each version, with identical prompt bytes and request payloads, exact output
counts, matching accounting, and zero preemptions. No per-cell 5% or weighted
2% investigation threshold is triggered. Relative to 0.31:

| Metric | 0.31 | 0.33 | Change |
|---|---:|---:|---:|
| Sum of measured wave times | 2501.51 s | 2508.10 s | +0.264% |
| Output tokens / measured wave time | 50.760 tok/s | 50.626 tok/s | -0.263% |
| Weighted native decode | 45.132 tok/s | 45.228 tok/s | +0.215% |
| Weighted native prefill | 1421.71 tok/s | 1418.36 tok/s | -0.236% |
| Card energy | 450525 J | 451528 J | +0.223% |
| Prefix tokens reused | 168000 | 168000 | unchanged |

The aggregate includes concurrency and prefill and is not a single-stream
decode-rate claim. Native decode is weighted across individual request decode
intervals; aggregate wave decode is also retained in the JSON. The
[single-request medians](serving-table.md) show context dependence. At 128K,
prefill is 1267.0 vs 1265.6 tok/s, retaining the earlier layout correction.

All 124 sampled output hashes differ across versions despite fixed seeds.
This is separate from the teacher-forced numerical comparisons below. Draft
acceptance also changes, so small decode differences do not isolate a kernel
speed change. These are one-host paired runs, not proof of universal
equivalence or a statistically significant speedup. Raw-result hashes are
retained in [serving-provenance.json](serving-provenance.json).

## Completed numerical checks

[GPU preflight](gpu-preflight.json), [kernel results](kernels/attention.json)
and [state replay](state-replay/) retain the numerical results and original
tolerances. The patched source boundaries and native identities are recorded
in [port-boundaries.json](port-boundaries.json) and
[runtime-identity.json](runtime-identity.json).

[Teacher-forced quality](quality.json) uses fresh paired captures. All 16,368
short-panel scored positions have byte-identical NLL/log-probability arrays.
For four long windows at 32K, 102K, 184K and 262K, suffix perplexity changes
by -0.054686%; full-prompt perplexity changes by +0.061910%. All 544 selected
full-vocabulary top-1 comparisons agree. This is finite distribution and
numerical evidence, not a task-accuracy evaluation or a claim that sampled
generated output is identical.

## Patch scope and reproduction

The [experiment directory](../../experiments/vllm-033-upgrade/README.md) contains
the exact wheel digest, pinned dependency build, guarded source port,
controllers and comparison scripts. The previous native source archive remains
the native bootstrap; the Python port is an additional explicit build stage.
Cached reconstruction reproduced the exact candidate runtime image; it does
not prove that a cold dependency resolution is deterministic.

The mmap-selection override is removed because upstream now supplies it.
The two CPU-offload loading queues remain bounded independently. GDN device
mask handling, EXL3 loading, MTP policy/pruned head, guarded oneDNN/M04 attention
and the production RAM/admission middleware still require local integration.
The fused SYCL GDN operator was already used by the qualified 0.31 engine;
0.33 adds backend selection rather than introducing that operator here.

The independent [0.31 release archive](../../../config/releases/exl3-v4/README.md)
retains its image, policy and original runtime modules. [Live rollback](live-rollback.json) is validated. Permanent-service deployment
passed; [post-deployment checks](post-deployment.json) verify the canonical
runtime hashes, release receipts and fresh loader audit.

## Operational result and deployment

[Operational evidence](operations.json) covers all 12 groups: actual rendered
tokenizer equality, API/streaming/tools/reasoning, images/videos and limits,
261120 input + 1024 output = 262144 total tokens, C4 and C16 long-request
accounting, large image context, disconnect/recovery, structured JSON and a
forced named tool, 32 distinct 4194304-pixel images, natural CPU-cache restore,
and completion-reserved admission for 200023- and 131020-token requests.
The latter queued safely with zero preemptions. C4/C16 also had zero
preemptions in these runs; this is not a universal no-preemption guarantee.

The natural restore transferred 1772355584 bytes from CPU to GPU after a large
competing history displaced the original GPU cache. Markers and the answer
remained correct. This is actual CPU restoration rather than only a warm
GPU-prefix hit. Abort testing checks client disconnect and subsequent
recovery, not an engine-side finish-reason trace.

The permanent user service uses image
`sha256:f1822d3b533eb77a3d9f9c3b610bab0c8334c329a40d6ed32041345d79ad567e`,
alias `local/b70-qwen38-vllm:production-exl3-v5`, with the canonical `runtime/`
modules and attested 12 GiB/LBNHC overlay. The promotion helper preserves the
previous config, modules and private environment, checks the strict launcher,
then verifies real text, SSE usage/default reasoning and a complete permanent
service restart. The new loader audit contains all 409 expected modules and
all eight MTP modules. Post-deployment host tests pass (38 passed, three
image-dependent skips); device numerical gates were run separately.

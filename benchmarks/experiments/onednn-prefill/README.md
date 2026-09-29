# FP8 KV to oneDNN Graph SDPA: feasibility screen

Status: **experimental image and operator proof of concept**, not promoted to production.
The code and graph construction adapt the EXL3 technique from `0xSero/exl3xpu`
(MIT license, copyright 2026 0xSero). The source and model formats remain GPTQ.

The donor's compiled operator cannot be used as a drop-in library with the pinned
Torch 2.13/vLLM 0.30 target. `native_sdpa.cpp` is built against that Torch ABI,
oneDNN 3.13, and the oneAPI 2026.1 compiler. `build_native.sh` deletes old
output before compilation and fails if no fresh library is produced.

## Initial feasibility evidence

Same synthetic FP8 cache operands, 1664-token pages, fragmented logical page
table, K scale 0.75, V scale 1.25, FP16 Q/output, 24 query heads, four KV
heads, dimension 256. Both arms run inside the measured W4A8 candidate image,
with Q128 as the current attention reference. Times are medians of three warm
XPU event samples and include the original Python gather/dequant, Q packing,
attention and output copy for the candidate. The native graph profiler reports four `micro_sdpa`
kernels, one per KV head. `torch.nn.functional.scaled_dot_product_attention`
with a lower-right bias was also tested; it dispatched to the unfused math
path and is not the candidate.

| Q | L | Q128 ms | Native graph complete ms | Candidate first call ms | Relative L2 error |
|--:|--:|--:|--:|--:|--:|
| 256 | 4,096 | 1.39 | 2.12 | 800 | 0.000141 |
| 4,096 | 32,768 | 79.07 | 40.29 | 591 | 0.000093 |
| 6,656 | 131,072 | 596.70 | 277.62 | 796 | 0.000071 |
| 6,656 | 200,704 | 926.77 | 431.06 | 1,000 | 0.000067 |

All outputs were finite and passed `torch.allclose(rtol=.01, atol=.002)`
against Q128. The first-call number includes allocation and graph compilation;
it is not an estimate of a production TTFT. The later native-gather and model
results are below. Concurrency, graph capture, and measured peak memory still
require qualification.

## Reproduce

Run `./build_native.sh` from this directory. Then run the script in the
measured image with the repository directory mounted as `/probe` and the
oneDNN install mounted as `/dnnl`; include `/dnnl/lib` in `LD_LIBRARY_PATH`.
For example, pass `--native --q 4096 --l 32768 --repeats 3` to
`sdpa_feasibility.py`. The `--inspect-kernel` option records profiler kernel
names in a separate diagnostic run. The binary is ignored by Git; the native
source and build procedure are the reproducible artifact.

Nonmultiple Q/L masks, fragmented pages, real attention operands, and
non-unit scales were tested in the follow-up screen. The build script and
adapter remain experimental.

## Serving and quality screen

An opt-in adapter in `docker/onednn-prefill/` preserves Q128, M04, and native
fallback. It was tested with the W4A8 candidate image, the pinned GPTQ model,
MTP4, FP8 KV, 200704 max context, 6656-token scheduler budget, and 180 W.
The reference flag stays off by default. Concurrent serving and full context
capacity are not yet qualified for the new route.

The broad long-prefill route improved median 32K native prefill from 1561.9
to 1793.3 tokens/s and wall time from 38.36 to 34.63 s across three paired
prompts. It **failed** the pre-existing teacher-forced NLL gate: changes were
+0.10767, -0.02108, and +0.03766 nats/token, mean +0.04141. Its sampled
output hashes differed from the control. Do not promote this broad route.

The broad route remains available as an explicit **performance-profile
candidate** with `B70_ONEDNN_PREFILL=1 B70_ONEDNN_PROFILE=performance`. The
reference profile remains the default and retains the frozen NLL limits.
The performance profile changes the default route to all eligible long
chunks through 200,704 KV tokens; it does not change the exact mask, page,
scale, or fallback requirements. Its NLL exceedance is a diagnostic warning,
not an automatic claim of acceptable task quality. No production promotion
is implied by choosing the profile. Run paired functional code tasks, source
review, long-context retrieval, and structured-output checks before judging
whether its extra speed compensates for any practical quality loss; the
first paired screen appears below. Retain
the old NLL bounds as the reference-profile gate; do not relabel a failing
candidate as passing that gate.
Moving the first full oneDNN chunk later, to KV length 26,624, still failed:
NLL changes were +0.01950, -0.00889, +0.05110, mean +0.02057. The frozen
limits are at most +0.02 per prompt and +0.01 on average. The broad route
therefore remains opt-in through `B70_ONEDNN_PROFILE=performance` (which
sets `B70_ONEDNN_SHORT_CHUNK_ONLY=0` by default).
Real-operand diagnostics found finite, `allclose` attention outputs with
relative L2 around 7e-5; per-token logprobs first diverged at token 13312,
the first chunk using the new route. Small local operator errors can therefore
have a material long-context model effect.

The paired per-token NLL analyzer now verifies prompt hash, token-ID hash,
scored positions, cache-hit count, and preemptions before reporting whole-
prompt and post-dispatch windows. On the first 32K development prompt, the
Q128 control and performance profile were identical before position 13,312.
From that position through the end, the performance profile worsened mean
NLL by **+0.18155 nats/token**, a 1.1991 perplexity ratio; whole-prompt
change was +0.10767. On the second development prompt, the post-dispatch
change was -0.03555 nats/token, while whole-prompt change was -0.02108.
These opposite outcomes prevent a universal interpretation of the mean.
Two additional 32K prompts, unused when tuning the route, changed NLL by
+0.02856 and -0.02478 nats/token under the broad route, relative to Q128.
The corresponding perplexity ratios were 1.0290 and 0.9755. A 16K pair
was an explicit negative control: its active KV length never reached the
16,384-token dispatch minimum, so both arms had identical NLL. It cannot
support a broad-route quality claim.

The selected-real-operand FP32 diagnostic used the exact visible KV prefix,
dequantized to FP16, then FP32 QK/softmax/PV. Across eight sampled attention
records from the first 32K prompt, median relative L2 error against this
reference was about 0.000222 for oneDNN and 0.000221 for Q128. Q128 was
closer in six records and oneDNN in two. These rows provide no sign of a
gross mask or page-layout error in the sampled call, but do not prove all
rows, heads, contexts, or concurrent executions correct. The diagnostic
does not run in scored timing measurements.

An additional cache-isolation check exposed a vLLM AOT-artifact confounder:
reusing a compile-cache directory across W4A8 and W4A16 flags produced
W4A16-labeled runs with W4A8 graph artifacts and W4A8-identical NLL.
With fresh separate caches, the W4A8 graph contained `int4_gemm_w4a8`
while the W4A16 graph contained `int4_gemm_w4a16`. On the first 32K prompt,
their Q128 NLL values were 3.14749 and 4.67535, respectively. The latter
also restored the slower archived W4A16 prefill range in a targeted serving
replay. The launcher now isolates experimental compile caches by image ID
and runtime arm. Any earlier cross-arm W4A16 result from a shared cache
must be discarded; within-W4A8 oneDNN comparisons used the same W4A8
graph and remain useful. A fresh-cache run of the exact archived control
image `sha256:648132c9...` reproduced the isolated W4A16 NLL 4.67535
exactly, with the same token IDs and scored positions. A fresh-cache run of
the current production tag (`sha256:2d289fbe...`) gave 4.65996. This
confirms the large single-prompt W4A8/Q128 versus W4A16/Q128 NLL gap is
not an artifact of the experimental oneDNN image. It does not establish
that W4A8 improves general model quality; more prompts, real activations,
and functional tasks are needed for that attribution.

The first practical performance-profile screen used 30 frozen, paired tasks
across five distinct 32K source contexts: ten executable Python coding tasks,
ten exact source retrievals, five snippet reviews, and five structured-output
tasks. Each arm began with an empty prefix cache and used the same image,
model revision, prompts, deterministic sampling settings, and task order.
The two review prompts with page numbering explicitly define zero-based
pages; the ambiguous pilot was excluded and both arms rerun from a fresh
server. All 30 requests finished normally on both arms, with identical
computed-prefill and cache-hit token counts and zero preemptions.

| Paired 32K task screen | Reference profile | Performance profile |
|---|---:|---:|
| Passed / 30 | 28 | 29 |
| Coding / 10 | 9 | 10 |
| Retrieval / 10 | 10 | 10 |
| Review / 5 | 5 | 5 |
| Structured / 5 | 4 | 4 |
| Sum of wall time, s | 173.95 | 164.45 |
| Sum of prefill time, s | 144.65 | 135.53 |
| Sum of decode time, s | 28.18 | 27.79 |
| Median cold 32K TTFT, s | 20.03 | 18.19 |
| Sum of five cold-prefill times, s | 100.32 | 91.22 |
| Sum of 25 warm-prefill times, s | 44.32 | 44.31 |

Seventeen of 30 output hashes matched. The one improved coding case passed
its executable examples on the performance arm; the reference arm emitted a
nonterminating loop. Both arms failed the same numerical page-remainder
task. The task count is too small to rule out a modest or task-specific
regression, especially at 128K/199K, under concurrency, or on less
structured coding requests. The observed 9.50-second sum-of-request-time
gain is dominated by the five cold 32K prefills; warm prefix reuse leaves
little room for this route to help. Decode totals include different generated
outputs and MTP acceptance, so they do not isolate decode kernel speed.
The broad route remains an opt-in performance candidate pending wider
quality and serving checks. The frozen tasks, raw responses and per-request
timing are in `performance_tasks.py` and `practical-*.jsonl`.

A second paired screen used two frozen 128K source contexts and 12 tasks of
the same four kinds. Both profiles passed all 12, including four executable
coding tasks and four exact retrievals. Nine output hashes matched. Prompt
hashes, image ID, computed tokens, and prefix-hit tokens matched pairwise;
there were no preemptions. The performance profile reduced summed request
wall time from 390.23 to 263.59 s. Two cold-prefill times summed to 325.74
versus 212.74 s (median cold TTFT 163.07 versus 106.57 s); ten warm-prefill
times summed to 47.15 versus 36.87 s. At 128K, several warm prompts lie
just above the reference profile's 131,072-KV-token cap and fall back to
Q128, while the performance profile uses oneDNN. Output lengths differed,
so the decode-time sums cannot isolate decode speed. This is a strong speed
result on two contexts, not evidence of broad 128K task-quality equivalence.
Raw responses and paired timing are in `practical-128k-*.jsonl`.

A third practical screen used one frozen near-maximum 199K source context and
six tasks: two executable coding tasks, two exact retrievals, one snippet
review, and one structured-output task. The reference profile passed 6/6;
the broad performance profile passed 5/6. Both passed the coding, retrieval,
and structured tasks, but the performance profile labeled a defined
zero-based page-numbering off-by-one bug as `zero_division`. The same exact
review prompt failed again on a warm-cache repeat. Image ID, prompt hashes,
tokenized lengths, computed-prefill and cache-hit counts matched pairwise;
there were no preemptions. Both arms had two cold 199K requests followed by
four warm-prefix requests. The broad profile reduced summed wall time from
721.55 to 427.10 s and the two cold-prefill times from about 672.56 to
398.76 s. Its review failure blocks promotion at this length even though
the speed gain is large. The data and per-task outputs are in
`practical-199k-*.jsonl` and `practical-199k-summary.json`.

On the same frozen 199K teacher-forced prompt, reference NLL was 2.16401345
and broad-performance NLL was 2.15622920 (delta -0.00778425 nats/token).
All 199,661 scored positions and token IDs matched, without cache hits or
preemptions. The first 13,311 scored positions were identical, and some later
windows worsened despite the better whole-prompt mean. This case shows why
aggregate NLL alone cannot overrule the observed task regression. See
`nll-199k-*-paired-r1.json` and `nll-windows-199k-performance-r1.json`.

A targeted 199K review experiment capped performance-profile oneDNN at KV
length 196,608, making only the final two prompt chunks use Q128. Route logs
confirmed oneDNN through L=193,024 and Q128 at L=198,016 and 199,717. The
formerly failing review returned `off_by_one` on both a cold request and a
warm-prefix repeat. The cold prefill was 207.47 s versus about 199.37 s for
the broad performance profile and 336.35 s for the reference profile's
prompt-matched cold request. A fresh-start replay of all six paired tasks
passed 6/6, including both executable code tasks and the review, with no
preemptions and identical per-task prompt/prefix counts. The two cold
prefills took 411.95 s combined versus 672.56 s for the reference and
398.76 s for unrestricted performance. Summed six-request wall time was
458.18 s versus 721.55 s for reference and 427.10 s for unrestricted
performance. The four warm-prefix prefill times summed to 39.82 s, about
the reference profile's 39.78 s: the late Q128 fallback gives up most of
the broad route's warm-prefix gain. Different outputs and token counts make
decode-time sums unsuitable as a kernel-speed comparison. This cap is the
better observed 199K performance candidate, but six tasks on one source
context do not establish general code/review quality or serving safety.
The targeted responses are in
`practical-199k-review-max196608*.json`.
The full replay is in `practical-199k-max196608.jsonl` and
`practical-199k-max196608-summary.json`.

On a third frozen 128K context unused in route selection or the practical
screen, teacher-forced NLL was 1.35621671 for reference and 1.34465623 for
performance (delta -0.01156048 nats/token). Token IDs and all 131,053
scored positions matched; neither arm had a cache hit or preemption. The
first 13,311 scored positions were identical. From position 13,312 onward,
mean delta was -0.01286742; however the later 65K-token half worsened by
about +0.00227. The whole-prompt improvement therefore does not justify a
universal NLL or task-quality claim. Compact summaries and token-window
analysis are in `nll-heldout-128k-*.json` and
`nll-windows-128k-performance-r3.json`.

One fresh-start staggered pair put a 4K request into 1,024-token decode,
then admitted a cold 32K request. Both profiles emitted 1,024 tokens from
each request, with no cache hits or preemptions. Reference versus performance
long-request TTFT was 21.65 versus 21.59 s; the largest observed gap between
streamed pieces of the already-decoding request was 3.99 versus 3.98 s.
Total measured prefill was 24.44 versus 24.37 s. The broad route showed no
material benefit in this mixed schedule. A stream piece can contain more
than one token under MTP, so this is a bundle-gap measurement, not token
interarrival latency. The one pair does not characterize p95/p99 stalls or
all concurrency patterns; the exact mixed-metadata route still needs a
dedicated trace before enabling a new attention arm there. Raw timestamps
and response hashes are in `staggered-*.json`.

In a fresh performance-profile server, a 128K prefill was cancelled after
30 seconds, after oneDNN had dispatched through at least KV length 59,904.
The server stayed healthy. Subsequent cold 4K and 32K recovery requests each
emitted 128 tokens, with no prompt-cache hits or preemptions; the 32K request
used the oneDNN route. This is one cancellation/recovery smoke test, not a
stress or leak test. The request and route evidence is in
`cancel-recovery-performance.json` and
`cancel-route-signatures-performance.txt`.

The narrower `B70_ONEDNN_SHORT_CHUNK_ONLY=1` policy is the default when the
experimental oneDNN flag uses the reference profile. It selects only an eligible
single-request chunk with fewer than 6656 query rows and active KV length
between 16,384 and 131,072 tokens. The upper bound keeps the 199K case on
Q128 after its sampled decode penalty; `B70_ONEDNN_MAX_KV` can override it
for explicit experiments. The performance profile sets short-chunk-only to
`0` as an opt-in evaluation of the NLL-failing broad path. The historical
`B70_ONEDNN_FINAL_CHUNK_ONLY=1` flag is an alias for reproducing earlier runs.
It is **not** a final-chunk detector: a 32K teacher-forced request dispatched
at both `(Q,L)=(3328,29952)` and `(2759,32711)`, so the first dispatch was
still an intermediate prompt chunk. The route cannot infer the remaining
prompt length from the current attention-call metadata and is not promoted.

| C1 context | Comparison | Q128/W4A8 | Short-chunk candidate |
|---|---|---:|---:|
| 32K | Median prefill tok/s | 1561.9 | 1636.2 |
| 32K | Median TTFT s | 20.99 | 20.04 |
| 32K | Median wall s | 38.36 | 36.81 |
| 128K | Median prefill tok/s | 797.3 | 813.9 |
| 128K | Median TTFT s | 164.55 | 161.17 |
| 128K | Median wall s | 190.13 | 186.54 |

The 32K control was replayed with the same derived image, attention flag off,
and reproduced all three archived W4A8 output hashes. The 128K comparison
uses the archived W4A8 control and prompt-matched candidate runs. All scored
requests produced 1024 output tokens, had no prompt-cache hits and zero
preemptions; sampled output hashes differ. The 32K short-chunk teacher-forced
NLL changes were -0.01306, -0.00881, -0.02663 nats/token and passed the
pre-existing +0.01 mean/+0.02 each upper bounds. One paired 128K prompt had
a -0.000067 nats/token NLL change. These limited results do not qualify
general quality, concurrency, prefix reuse, or maximum context.

The fused SYCL page gather removes the FP8/FP32 Torch intermediate chain and
matches the Torch gather oracle bit-for-bit at page boundaries with non-unit
scales. The native graph now uses the caller's in-order SYCL queue, persistent
host mask lengths, an allocator tied to the XPU caching allocator, and a
32-shape per-context LRU cache with event-aware eviction. Its compiled
`micro_sdpa` scratchpad reports zero bytes at the measured long shapes.
For Q=4096/L=32768, complete adapter median was 37.41 ms versus Q128 79.58;
for Q=6656/L=200704 it was 413.82 versus 925.74 ms. In three frozen 32K
serving replays, median prefill was 1661.0 versus 1561.9 tokens/s, TTFT
19.75 versus 20.99 s, and wall 36.50 versus 38.36 s. All three sampled
output hashes and all three teacher-forced NLL values matched the earlier
short-chunk prototype exactly. This validates the gather change for those
cases, not broader model quality or serving safety.

The same three frozen 64K prompts passed the teacher-forced NLL gate with
changes of -0.000766, -0.000704, and +0.003217 nats/token (mean +0.000582).
Neither arm had a prefix-cache hit or preemption. This does not establish
quality for untested prompts or near-maximum context.

For a fixed 1,024-token continuation at 32K, both arms emitted the same
allowed token and output hash on all three prompts. Median native prefill was
1,549.49 versus 1,653.06 tok/s, TTFT 21.198 versus 19.845 s, wall time
31.790 versus 30.474 s, and native decode time 10.601 versus 10.635 s
(control versus short-chunk candidate). The 0.034 s decode difference is
about 0.3% and does not establish a decode-kernel regression. Freely sampled
continuations can differ in MTP acceptance and therefore decode time.

On the rebuilt final image, one paired 32K fixed-continuation replay reused
29,952 prompt tokens in each warm arm and recomputed 2,770. The warm Q128
versus oneDNN result was TTFT 2.333 versus 1.767 s, wall 12.876 versus
12.365 s, and native decode time 10.550 versus 10.605 s. Output hashes were
identical, with 1,024 emitted tokens and no preemption. The oneDNN warm
decode difference is about 0.5% in one pair; it is too small and too sparsely
sampled to claim a decode-kernel regression. The cold final-image pair also
completed with identical output hashes and no cache hits; TTFT was 21.806
versus 20.577 s and decode time 10.549 versus 10.599 s.

A fresh-start 199K frozen serving request completed on each arm with 199,673
prompt tokens and 1,024 output tokens, no cache hit or preemption. The
candidate improved prefill from 594.97 to 612.11 tok/s and TTFT from
335.97 to 326.56 s, but freely sampled decode time grew from 29.28 to
33.05 s; acceptance fell from 672/1,404 to 630/1,584. Wall time fell from
365.21 to 359.57 s. This single request is a capacity smoke test, not a
safe 199K promotion. One same-image, teacher-forced 199K prompt had NLL
2.16401345 on Q128 and 2.16401628 on oneDNN (change +0.00000283
nats/token), with no cache hit or preemption; it passed the frozen numerical
bound but does not establish broad 199K quality. Keep this route opt-in while
199K cached-prefix decode behavior and wider quality coverage remain open.

## Follow-up gates

The context cache is now capped at four device/context pairs, with 32 compiled
Q/L shapes per pair. An event is awaited before an in-flight shape is evicted.
A 33-shape test observed 34 misses and two evictions after revisiting the
first shape; its output was bit-exact after recompilation. The cache counters
are printed only with `B70_ONEDNN_DIAGNOSTICS=1`.

At synthetic Q=6656/L=200704, the Torch XPU allocator reported 329 MB peak
extra active allocation for the full native adapter call versus 82 MB for
Q128. The difference of 247 MB includes the reusable K/V and Q/output
working buffers. This is an operator-level measurement, not full-server VRAM.

The pinned W4A8 operator passed finite and zero-row checks at 511/512/513
matrix rows with typical and outlier synthetic activations. Relative RMS
error against W4A16 was about 0.88% for typical rows and 1.33% with an
outlier. The current dispatch still uses total matrix rows, so a mixed batch
can quantize decode rows; no mixed-row quality claim is made.

C2 with two 16K requests and C4 with four 4K requests completed 1024 output
tokens per request with no preemptions; their mixed metadata used the existing
attention fallback. A 16K prefix replay reused 13,312 tokens. A 32K replay
used the oneDNN route cold, then reused 29,952 tokens warm without dispatching
oneDNN and without preemption. These are serving smoke tests, not latency
comparisons against a prompt-matched control.

A diagnostic attempt to change the oneDNN Graph softmax probability tensor
from FP16 to FP32 failed graph construction (`could not add op to the graph`).
The FP32-probability variant was not run as a model candidate. The Q128 kernel
uses a blocked online softmax with `exp2` and folds FP8 K/V scales into the
score/output arithmetic; the fused oneDNN path uses a different reduction
and FP16 K/V. That is a plausible source of numerical differences, not an
isolated cause for the 32K NLL regression. The real failing cases used unit
K/V scales, so scale multiplication alone cannot explain them.

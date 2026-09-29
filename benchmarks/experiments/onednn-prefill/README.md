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
Moving the first full oneDNN chunk later, to KV length 26,624, still failed:
NLL changes were +0.01950, -0.00889, +0.05110, mean +0.02057. The frozen
limits are at most +0.02 per prompt and +0.01 on average. The broad route
therefore requires an explicit `B70_ONEDNN_SHORT_CHUNK_ONLY=0` research flag.
Real-operand diagnostics found finite, `allclose` attention outputs with
relative L2 around 7e-5; per-token logprobs first diverged at token 13312,
the first chunk using the new route. Small local operator errors can therefore
have a material long-context model effect.

The narrower `B70_ONEDNN_SHORT_CHUNK_ONLY=1` policy is the default when the
experimental oneDNN flag is enabled. It selects only an eligible
single-request chunk with fewer than 6656 query rows and active KV length
between 16,384 and 131,072 tokens. The upper bound keeps the 199K case on
Q128 after its sampled decode penalty; `B70_ONEDNN_MAX_KV` can override it
for explicit experiments. Setting short-chunk-only to `0` is
reserved for explicit research on the quality-failing broad path. The historical
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

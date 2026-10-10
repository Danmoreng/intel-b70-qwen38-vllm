# E01 stable oneDNN maintenance — qualification in progress

Original loaded custom library ee8fc42... maps to clean local oneDNN source
0e2a5bfe..., original GPU SYCL/CPU NONE/Graph ON build. Its embedded micro.cl
retains initialization after the early barrier-arrive. Stable v3.13.4 source
c531fe1... has the corrected initialization before barrier-arrive. Audit and
source excerpts are retained in oneDNN-provenance-r2; this is no longer an
unresolved vendor/dirty-build assumption.

Stable library SHA256 `90c0feddfb51a6219903d5e1ccabc9fd32dab0c702284a16dd7692bb6042e056`; native EXL3 rebuild SHA256
`00b65587220481ec89269f7510136146809d7e01938de3d30d5b67206948a95a`.
Torch 2.14.0+xpu, SYCL9/oneAPI26.1.1, native flags and EXL3 source commit
fc8f8c9... retained. Candidate image b772d379... is an experimental child of
the exact qualified image, preserving installed MTP policy and source guards.

Artifact registration/import, 31 attention numerical cases, 3 causal poison
cases, 2 mutated graph suites and real-weight W8A8 prefill reference PASS.
Ordinary creation logging confirms GPU SDPA `ocl:micro:reusable` and fused
`sdp_primitive_kernel_t` on FP16 Q=256/K=4097/GQA6/D256, strict fpmath.
No execution verbose profiler was requested from the queue.

81-key, three-epoch application cache soak: capacity64, peak64, 177 evictions,
488 hits. Post-warmup growth: RSS 749568
bytes, GPU allocated/reserved both zero growth. This finite soak is not a
universal proof about all driver/oneDNN allocations or in-flight pressure;
pressure_waits was zero. The exact 261120-input + 1024-output capacity request completed without
preemption. The first three paired serving blocks are effectively neutral
(cold weighted decode +0.108%, warm +0.021%, both uncertainty intervals cross
zero). No release promotion or decode-speedup claim yet.

The exact short-path array comparison failed and remains a failed raw receipt.
A separate changed-stack numerical review passed after the original-tolerance
126 real-weight linear cases, four actual-state attention replays, 12 difficult
FP16 SDPA cases, stored BF16 quality references evaluated on the CPU, and bounded
same-prefix D64/rejection checks. Short and long quality panels are mixed or
neutral; their results are not a general quality-improvement claim. See the
linked JSON reviews, including the retained failure and its limitations.

A controlled accounting pair with empty compiler caches on both images
reproduced the same 199-block pool and approximately 21-second first source32
prefill. Ordinary subsequent starts use the previous 200-block pool; no KV
memory/utilization override was introduced. Runtime registered tensors show
no BF16 in either target or MTP. Storage BF16 reference files do not establish
a BF16 execution bridge. See E01-firsttouch-accounting.json and E01-memory-ledger.json.

Three more independent paired blocks now include short C1, 128K C1 and the
exact source32 fixture at C4. All previous measurements remain included.
Only a reviewed non-regressing maintenance survivor can proceed to immutable
metadata, operational gates and the full frozen final matrix. Production
remains the original qualified image after every controller.

Six core paired blocks now show weighted cold decode +0.659% [0.102,1.285]
and warm -0.681% [-1.823,0.316]. Neither meets the performance gate. Prose-C4
warm native decode averages -1.882%; its counter decomposition is -1.630%
usable tokens per request round and +0.256% request-round equivalent time.
These are an exact algebraic decomposition of overlapping request counters,
not causal kernel attribution or GPU group-round latency. Client aggregate
prose-C4 warm averages -2.824% with wide paired uncertainty [-5.82,0.05].
See the unfiltered six-block report and counter decomposition.

Two further unchanged protected blocks are running to resolve the remaining
C4 concern and bring the protected cells to five independent paired starts.
No release review is approved from the six-block file alone.

Eight unfiltered core blocks and five protected blocks are complete. Weighted
cold decode +0.111% [-0.701,0.893], warm -0.752% [-1.865,0.236]. Warm prose-C4
native -1.776%, client aggregate -2.464% [-5.363,0.132] remain adverse results.
The registered no-credible-loss screening rule passes, but the wide interval
is not a tight-margin noninferiority proof. This is maintenance-screen approval,
not a performance or production promotion.

Final metadata image `36e41ea...` preserves the exact tested parent RootFS
`b772d379...`; new policy and requirements explicitly pin oneDNN 3.13.4 and
the rebuilt EXL3 library. The original v3 rollback passes strict preflight.
Exact-image operational gates and then the complete frozen serving matrix
are now running. See E01-maintenance-serving-review.json for the full tradeoff
and limits. Do not infer final approval from this progress report.

The exact-image operations are now complete and passed. The original C16
comparison failure (3 versus an archived 2) remains failed. Three named
contemporaneous reference pairs recorded baseline 2/3/3 and candidate 2/2/2,
all without extra prefill work; the separate reviewed reference is explicit.
The final operations run also recorded two, matching the old count. Actual
incoming-prefill/decode overlap passed at two and four admitted requests,
32 distinct maximum-area images used 131164 input tokens without preemption,
16 active source guards passed, abort was confirmed and the worker restarted
with the original 200-block pool. The strict exact-image launcher preflight
passed. The full 70-wave/124-request serving matrix is running; there is still
no production promotion or claimed speedup.

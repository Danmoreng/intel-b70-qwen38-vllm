# Phase A findings and next experiment

Production remains the qualified MTP3 image, FP16, 262144 tokens and 180 W.
These are diagnostic results, not a production performance qualification.

## Actual proposal law

The existing B8/V7 launch omitted `draft_sample_method`, whose vLLM 0.31 default
is `greedy`. The speculator therefore has `draft_logits=None`; the selector
chooses argmax and ordinary rejection uses a Dirac proposal. Softmaxing its
realized scores would describe a proposal distribution that was never drawn.
This is not evidence of incorrect target sampling. The review's probabilistic
q-cache assumptions apply only after explicitly selecting probabilistic drafts.

Eight selected actual rounds were captured from 116 decode rounds using the
frozen 32K prompt, T=1.0, top-p=.95, top-k=20, seed=770000. CPU FP64 auditing
confirms the Dirac overlap and normalized positive residual identity. Target
sampling transformations are captured before conversion to p.

Independent statistical proposal/rejection tests cover four exact distributions
at T=1.0 and 1.95, 32768 samples per case, familywise alpha .001. Maximum bin
error .00505 is below the predeclared simultaneous bound .01347. Seven-step
Top16 graph replay checks conditional predecessor indexing, exact realized
score caching, candidate eviction and permuted persistent request slots.
Synthetic paged draft attention matches an independent FP32 dense oracle for
the same stored FP8 and FP16 K/V: GQA4, head dimension128, B8, noncausal symmetric
2047-token windows, page64/1664, SWA2047/2048/2049, C1/C4 and stale tail slots.

## Cost split

116 pure-decode event spans at unchanged B8/V7 yield medians:

| Stage | Diagnostic GPU timeline span, ms |
|---|---:|
| Whole runner round | 62.28 |
| Target body | 48.08 |
| Target full head | 2.05 |
| Verification sampling | .68 |
| Complete draft proposal | 11.02 |

These spans include dispatch gaps; medians must not be added as exact totals.
An eight-round separate XPU trace records real kernel durations. Dense GEMM
symbols do not yet uniquely separate feature-FC, context-KV and draft backbone.
Kineto graph annotation envelopes overlap at some cycle boundaries, so ambiguous
kernel attribution is retained rather than guessed. Source and runtime hashes,
raw traces and tensor fixture hashes are preserved.

Complete elimination of the draft would remove at most approximately 17.7% of
the measured round span (ideal speedup1.215). Practical weight quantization can
remove only part of this. There is no case for a large selector rewrite from
these measurements.

## Memory

The actual current worker has432 blocks, 17039360 bytes per shared pool block,
6.855GiB unique KV storage. Pool descriptors overlap and must not be summed.
The previous369-block startup differs in available profiled memory; it is not
a fixed hardware capacity. Required blocks remain330 for96K and562 for192K.
Thus the unchanged192K CPU model needs8.918GiB. A group-size-four model
(4 full,12 Mamba,2 draft-SWA groups) needs588 smaller blocks,7.465GiB. Neither
calculation qualifies actual192K serving.

Unmarked DFlash groups cause `KVCacheCoordinator` to conservatively apply the
EAGLE last-block-drop treatment to all groups. Do not change markers without
prefix/rejection boundary gates; this alone does not explain acceptance.

## Decision

First test probabilistic drafts as a single changed variable at B8/V7. Existing
kernel gates support a diagnostic experiment; full runtime q capture is still
required. Next integrate the published W4/G128 draft using the existing XPU
oneDNN W4A16 operator, with an explicit draft namespace and strict independent
inventory. Target EXL3 and the shared full head remain unchanged. Quantization
and any group-size change are separate arms. B8/V4 follows only if these results
leave an evidence-backed reason to reduce verification cost.

Published W4 source: [syvai model card](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16),
revision4d30ec736ffc6b8688dc2ae2b502d9b48bdec279, weights SHA256
ec26996e6a0745ab5edb857117220ce1e219ad524f71e6e149b703804947d8e7.
It quantizes FC, attention and MLP, preserving convolution projections and
selector. Dense BF16 tensors will be converted offline to FP16. Its NVIDIA
measurements do not establish B70 performance.

## Remaining gates and exclusions

Graph-internal actual model branches/unary/selector edges and selected actual
KV projections are not yet captured; runtime RoPE vs independent RoPE and
EOS/context-limit state gates remain open. No DFlash promotion is authorized by
this partial diagnostic status. The initial capture script used the wrong
sampling values; its raw run is excluded. A subsequent initial sampler fixture
put the proposal in the anchor slot and failed; that fixture ABI was corrected
and independent probes passed. Raw failures remain preserved. Every GPU test
restored and verified the exact qualified production image.

For the later bounded MTP3/4 decision, predeclare workload weights35%32K-C1,
35%64K-C1 and30%prose-C4 as an explicit working assumption. Cold/warm prefix
states are separate reported strata. Require>=5% weighted improvement with a
positive paired uncertainty interval and no important C4 latency regression.

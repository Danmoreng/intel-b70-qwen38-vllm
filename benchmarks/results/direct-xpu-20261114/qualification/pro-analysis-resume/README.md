# Packed M04 attention: cause and correction

The independent review identified two gaps in earlier diagnostics: synthetic
sampler probes exercised inactive V1 code, while the worker uses V2; and a
parsed-text repetition counter missed loops inside incomplete tool calls.
The corrected helper measures raw token runs. HTTP completion alone does not
establish a valid response.

## First invalid operation

All 96 installed GPU-runner Python files match the recorded R11 inventory.
The active V2 boundary trace reproduces all 2,048 failing output IDs exactly,
with 1,521 V2 rejection calls and zero legacy calls. At step 372, before output
index 885, hidden states and raw logits become invalid. This boundary trace
alone does not locate the internal producer.

A subsequent narrow-window layer trace also reproduces every output ID. It
finds the first invalid output in decoder layer 3's attention, at positions
2,302–2,305: inputs are finite, but the first two query rows become NaN.
This places the observed error inside model attention, before the LM head
and sampling. See the [boundary](first-invalid-boundary.json),
[layer](first-invalid-layer.json) and [active-path identity](active-v2-identity.json)
receipts. Instrumentation adds device kernels and may perturb execution;
it is used only in diagnostic images, never for performance measurements.

## Why the new headers exposed the bug

The newer XPU attention kernel normalizes each split independently and then
combines the partial results. Our packed causal mask can leave a split fully
masked for an early query row while later rows use it. That legal empty
split has zero denominator: normalizing it produces `0/0`, and its later zero
weight does not remove NaN. The previous kernel kept split outputs
unnormalized and skipped zero-mass splits.

The [minimal patch](../../../../../engine/direct-xpu/m04-empty-split.patch)
emits the mathematically correct zero contribution for an empty causal split.
Nonempty normalization is unchanged. It is limited to our packed-mask kernel;
it does not sanitize logits, suppress tokens or change sampling.

| Matched control | Result |
| --- | --- |
| Independent M04 probe, original new headers | 11 of 40 cases fail |
| Same probe with the empty-split correction | 40 of 40 pass |
| Captured real request with correction | Complete tool call; no repetition loop |

The real request retains exactly the first 885 tokens of the failing arm.
The standalone probe compares actual FP8-KV GPU attention against an independent
CPU FP32 calculation, with the original tolerances. See the
[cause receipt](m04-cause-and-fix.json), [unpatched](m04-boundary-unpatched-r1.json),
[patched](m04-boundary-patched-r1.json) and [request](m04-fixed-raw-result.json)
results. These diagnostics establish the local cause and correction; they do
not substitute for full release qualification.

The clean R12 candidate restores standard compilation and retains MTP3, native
`FULL_DECODE_ONLY` graphs and the newer libraries. Its installed GPU-runner
sources contain no diagnostic instrumentation. The earlier Inductor workaround
was based on incomplete observations and is withdrawn. This failure is not
evidence of a PyTorch 2.15 arithmetic defect. The separate PTI graph-lifetime
backport remains necessary and is documented in the
[qualification report](../../README.md).

Full diagnostic recipes, patches, raw responses and traces remain in the local
`benchmark-results/direct-xpu-20261114/pro-analysis-resume` directory. Current
final-image test status is in the [qualification report](../../README.md).

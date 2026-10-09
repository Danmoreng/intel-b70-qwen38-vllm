# DFlash2 FP16 residual overflow experiment — vLLM 0.31.0

This isolated experiment fixes the DFlash2 residual-add overflow reported in
[vLLM #55250](https://github.com/vllm-project/vllm/issues/55250).
The related [PR #55294](https://github.com/vllm-project/vllm/pull/55294) proposes
rejecting FP16; it does not implement this numerical fix. Both were open when
reviewed on 2026-10-09. The installed 0.31.0 source still has the unscaled path.

The qualified production image and MTP3 profile remain unchanged. GPU experiments
hold the shared benchmark lock and restore the qualified service after each run.
There is no DFlash throughput or serving acceptance claim in this experiment.

## Numerical change

FP16 stores finite values up to 65,504. The report identifies a fused residual
addition overflowing in draft layer 1. Keep the decoder residual stream and
convolution finish outputs in units of `a = 1/16`, and scale the input,
post-attention and final decoder RMSNorm epsilon by `a*a`. In real arithmetic,

```
RMSNorm(a*x, a*a*epsilon) = RMSNorm(x, epsilon)
```

Every branch still receives the ordinary normalized input. Scale the convolution
accumulator before storing the branch output in FP16. Scaling after an already
overflowing narrow store cannot repair it. The convolution uses FP32 internal
accumulation, as its existing CUDA implementation already does; its stored inputs,
weights, activations and residuals stay FP16. No clipping, NaN replacement or
runtime FP16/BF16 bridge is used. Context feature projection and context RMSNorm
are unchanged. Power-of-two scaling may affect subnormal rounding; it does not
prove bit identity or prevent arbitrary unbounded activations from overflowing.

The XPU FP16 convolution now uses the FP32-accumulating Triton kernel. Canonicalize
noncontiguous input views because the upstream kernel assumes unit inner strides.
CPU FP16 fallback also accumulates before narrowing. BF16 behavior is preserved.

## Reproduction

`Dockerfile` derives only from the exact qualified production image. `patch.py`
requires the SHA-256 of the reviewed upstream `qwen3_dflash2.py`, checks each edit
site, and parses the resulting Python source. Reusing this patch with a different
vLLM source requires a new review.

```
docker build -t local/b70-dflash2-fp16:v031-test benchmarks/experiments/dflash2-fp16-v031
python benchmarks/experiments/dflash2-fp16-v031/run_gpu.py --name numerics --script test_numerics.py
```

`test_numerics.py` checks the reported overflow against an independent FP32 norm,
small/zero/large inputs and the scaled epsilon, causal convolution boundaries,
strided views, block sizes 4/6/8, C1/C4, compile and XPU graph replay.

`capture_target.py` captures auxiliary states after target layers 5/19/33/47/61,
actual text positions, mask and sampled bonus embeddings. It explicitly runs the
EXL3 target in FP16 with the existing 262144 context limit and MTP3, in eager
mode for diagnostic hooks. Runner-marked padding rows are excluded. This is
input capture for the new draft experiment, not a repeat baseline benchmark.

`replay_draft.py` executes the patched vLLM decoder forward and grouped convolution
through all five layers using the pinned checkpoint and real target captures.
The unscaled baseline uses the same FP32 convolution accumulator as the patched
path, so it isolates residual overflow rather than measuring the original XPU
fallback's additional intermediate rounding. The final decoder RMSNorm also uses
the actual vLLM fused-add module. Its reference retains residual sums and convolution finish outputs in FP32 while
keeping linear/MLP/attention storage in FP16. Explicit standalone GQA/SWA/RoPE
attention is used for this numerical isolation; paged serving attention, FP8
cache encoding, full-vocabulary candidate generation and speculative acceptance
need separate integration qualification. The predeclared gates are finite output,
relative L2 below 0.02 and per-row cosine above 0.999.

Draft checkpoint: `incoai/Qwen3.8-27B-DFlash2`, revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`, SHA-256
`67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.
BF16 checkpoint weights are converted once on CPU to FP16 before device upload;
there are no BF16 tensors in the tested draft execution. The checkpoint contains
approximately 1.92 billion parameters, including the two candidate codebooks.

## Next integration gates

Measure actual added model/activation/graph/KV-group memory at the existing context
limit before throughput tests. The subsequent
[serving screen](../dflash2-serving-v031/README.md) records the user's explicit
authorization to lower context temporarily for unquantized speed experiments;
the eventual serving minimum remains 192000 tokens. Quantized FP16-output draft
linear layers are a separate candidate if necessary, with retained residual
scaling and new numerical gates.
Then verify draft noncausal attention, candidate selection, nonzero acceptance,
prefix reuse, outputs/abort/restart and long-context capacity before benchmarking.

## Verified results (2026-10-09)

[Public numerical receipts](../../results/dflash2-fp16-v031/summary.json) record:

- CPU and B70/XPU primitive gates passed, including compile and XPU graph replay.
- Real target captures: two prompts, 4420 and 3625 tokens, EXL3 FP16/MTP3,
  unchanged 262144 context limit. Draft SWA replay uses the actual 2048-token
  sliding window at the last anchor, plus a 512-token earlier anchor.
- Twelve real-weight cases across all five layers, block sizes 4/6/8: the
  unscaled residual baseline is nonfinite in 12/12; the scaled FP16 path is
  finite in 12/12. Maximum stored scaled residual: 9848, below 65504.
- Maximum relative L2 difference from the FP32-residual reference: 0.5632%;
  minimum per-row cosine: 0.99999499. These are numerical replay measurements,
  not a claim about token acceptance or exact output-token equivalence.
- Checkpoint parameter count: 1,924,404,480; FP16 parameter bytes: 3.5845 GiB.

Failed initial target-capture attempts are retained in local campaign evidence
and excluded from these results. Those attempts omitted an explicit target dtype
and resolved the diagnostic target to BF16. The successful capture explicitly
uses `dtype="float16"`; no failed capture is used by the replay.

The qualified production image remains `194a8786f120...`; the isolated candidate
is `41def154d401...`, with patched DFlash2 source SHA-256
`34da1d98d4f89c0b32ee26be95555c5098440d7d2d38931f6f913483dcac3e3e`.
The numerical fix is ready for full DFlash integration gates. It does not yet
qualify a DFlash serving profile, memory budget, quantization or throughput.

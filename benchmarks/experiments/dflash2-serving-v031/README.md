# Unquantized DFlash2 serving screen — B70 / vLLM 0.31.0

This experiment integrates the [FP16 residual fix](../dflash2-fp16-v031/README.md)
with the actual DFlash2 speculator, compiled serving, paged FP8 KV cache,
candidate selection and the EXL3 target. The user authorized lower context for
this unquantized speed screen. The eventual serving requirement remains at
least **192000 total tokens**. There is no quantized draft in this experiment.

## Model and isolation

- Draft: `incoai/Qwen3.8-27B-DFlash2`, pinned revision
  `dedf8df68adfb1afeaf7b7480c0a0243108177b4`.
- Unquantized parameters: **1,924,404,480**, **3.5845 GiB** in FP16.
  `prepare_fp16.py` converts the pinned BF16 checkpoint once on CPU to an
  offline FP16 checkpoint. Both target and draft execute in FP16, with no
  runtime BF16 bridge. The numerical fix retains FP32 accumulation inside
  the grouped-convolution kernel; its stored activations are FP16.
- Target: unchanged Qwen3.8-27B EXL3 4bpw body / 6bpw full head, FP8 KV,
  180 W, `gpu_memory_utilization=0.965`, admission 16, batched-token limit
  4096, production multimodal reservation and compilation/graph settings.
- Child image: `a24e00efd83df6fb944137223c9da0747c48cc105641aae62e021340b4486788`.
  Qualified production image: `194a8786f12065e65647d914ddc7fdc136bafbeec73f476730ef1324e4d2c7ec`.
- The strict EXL3 inventory audit now requires MTP checkpoint modules only
  when the speculative method is MTP. An external unquantized DFlash body
  shares the full target embedding/head and does not instantiate target MTP
  modules. `test_audit.py` also verifies rejection of missing target/MTP
  modules, missing MTP instantiation, duplicate target loads and unknown loads.
  This guarded patch is installed only in the experimental child image.

Every GPU campaign holds the shared benchmark lock, stops production, exposes
only a local test endpoint, and restores the exact qualified production image
in `finally`. A worker attestation checks actual draft/speculator classes,
FP16 dtypes, finite parameters, residual scaling/epsilon, absent draft body
quantization, shared full target head and absence of MTP vocabulary pruning.

## Capacity and speed method

The 192000-token startup attempt rejected capacity: **8.21 GiB** of KV memory
needed versus **5.86 GiB** available. vLLM estimated a maximum of **129600**;
that estimate is not a verified usable context limit. Model loading took
**19.53 GiB**, compared with **16.25 GiB** for the qualified target plus MTP3.
Removing MTP accounts for the difference between the net increase and the
3.5845-GiB draft. KV grouping, reserved activations and graph memory also affect
capacity; draft weight size alone does not determine context loss.

The experimental server successfully starts at **96000** tokens. The exact
context probe cycles token IDs from the frozen source prompt and requests
94976 input plus 1024 output tokens. It measures capacity, not semantic
long-context retrieval quality. This probe passed at depth 7 with exactly
96000 total tokens, a nonempty 1024-token output and no preemptions.

The speed screen uses the already frozen production fixture at nominal 4K, 32K
and 64K input budgets, C1, 1024 output tokens, three measured repeats per size and
one separate warmup. `compare.py` verifies byte-identical prompts and request
metadata against the matching first three repeats of the stored, qualified
vLLM 0.31 MTP3 benchmark. It does not repeat the old baseline. Decode is the
median native per-request weighted decode rate; prefill uses native compute
phase timers. These are not aggregate batch throughput or output tokens
divided by complete request wall time.
Actual prompt counts are 4083–4094, 32722–32762 and 65491–65532 tokens,
respectively, identical in the stored MTP3 and new DFlash requests.

## Native serving screen results (2026-10-09)

| Nominal input | Stored MTP3 | DFlash, 3 draft tokens | DFlash, 7 draft tokens |
| --- | ---: | ---: | ---: |
| 4K | 54.71 tok/s | 52.23 tok/s | 57.56 tok/s |
| 32K | 55.19 tok/s | 48.14 tok/s | 48.37 tok/s |
| 64K | 48.29 tok/s | 44.30 tok/s | 41.07 tok/s |

These are three-repeat decode medians from matching frozen requests. DFlash's
weighted accepted/drafted-token ratios at depth 3 are 55.5%, 55.7% and 54.3%;
at depth 7 they are 31.9%, 32.1% and 30.4%. Ratios at different draft lengths
do not directly measure speed: more proposed tokens can still yield more
accepted tokens per verification pass.

At depth 3, target verification fits the qualified M04 q2..5 guard. At depth 7,
the target's eight verification rows use the native attention fallback; this is
confirmed in worker dispatch logs. The subsequent
[M04 widening experiment](../m04-wide-verification-v031/README.md) investigates
that implementation limit before judging an optimized DFlash profile.

The production sampling policy is retained (temperature 1, top-p 0.95, top-k 20,
the frozen seeds, thinking disabled). Fixed-length output is a speed fixture,
not a semantic grade or proof of token-by-token equality. Separate smoke and
cold/warm prefix retrieval checks establish working nonzero acceptance and
correct retrieval for their limited fixtures. Abort, multimodal, stochastic
distribution and full production concurrency qualification remain separate.

## Reproduce

```sh
draft_source="$HOME/.cache/exl3xpu/incoai-Qwen3.8-27B-DFlash2-dedf8df6"
draft_fp16="$HOME/.cache/exl3xpu/incoai-Qwen3.8-27B-DFlash2-dedf8df6-fp16"
mkdir -p "$draft_fp16"
docker run --rm --network none --user "$(id -u):$(id -g)" --entrypoint python \
  -v "$draft_source:/source:ro" -v "$draft_fp16:/destination" \
  -v "$PWD/benchmarks/experiments/dflash2-serving-v031:/experiment:ro" \
  local/b70-dflash2-fp16:v031-pinned /experiment/prepare_fp16.py
docker build -t local/b70-dflash2-serving:v031 benchmarks/experiments/dflash2-serving-v031
python benchmarks/experiments/dflash2-serving-v031/run_serving.py --name unquantized-k3 --benchmark
python benchmarks/experiments/dflash2-serving-v031/run_serving.py --name unquantized-k7 --depth 7 --contexts 96000 --benchmark --capacity-probe
```

The build requires the exact preceding numerical-fix image and guarded audit
source. The CPU audit regression script is run inside this child image.
Raw local evidence resides under `benchmark-results/dflash2-serving-v031-20261009/`;
[public experiment receipts](../../results/dflash2-serving-v031/comparison.json)
retain source/result hashes and matching frozen prompt/request hashes.

The first completed depth-3 campaign had a controller postcheck field-name
error after all nine waves completed. The original failed receipt and executing
source are retained. `review_completed_screen.py` independently checks the
existing output counts, native measurement data, attestation, audit, smoke,
prefix checks and restored production image; it creates a separate reviewed
receipt and does not rerun GPU requests. One asynchronous sampled window has a
one-token accounting residual. Its round-equivalent metric is excluded; native
request-phase timing does not depend on that sampled counter window.

Production README benchmarks and historical power comparisons are unchanged.
These experiments do not replace the 262144-token MTP3 serving profile.

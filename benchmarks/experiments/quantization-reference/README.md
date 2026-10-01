# Qwen3.8-27B quantization reference panel

Compare the pinned BF16 original with the locally deployed GPTQ and EXL3 4-bpw
checkpoints using identical, teacher-forced token sequences. This is a short
distribution/likelihood diagnostic, not a generated-code quality benchmark.

## Measured result, 2026-10-01

Lower perplexity and forward KL are better. KL is measured against the BF16
original; top-1 agreement uses the same 512 sampled prediction positions.

| Checkpoint / precision | Mixed-panel PPL | PPL change vs original | Mean KL, nats | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| Original BF16 | 3.604529 | baseline | 0 | 100% |
| Original, FP16 control | 3.603800 | -0.0202% | 0.000458 | 99.22% |
| GPTQ Int4 G128, FP16 activations | 3.801927 | +5.4764% | 0.086369 | 93.36% |
| XReyRobert GPTQ-Pro FOEM Int4 G128, FP16 activations | 3.800345 | +5.4325% | 0.071485 | 92.97% |
| EXL3 4 bpw / 6-bpw head, FP16 activations | 3.641980 | +1.0390% | 0.031493 | 95.90% |

The code subset shows PPL increases of 9.32% for GPTQ and 1.66% for EXL3.
The prose subset shows increases of 1.77% and 0.42%, respectively. On this
panel, the native EXL3 checkpoint/runtime is closer to the original than the
native GPTQ checkpoint/runtime. The paired GPTQ-minus-EXL3 NLL difference is
0.04298 nats/token, with a window-bootstrap 95% interval of [0.02284, 0.06506].
This does not establish the cause of the earlier agent trajectory differences.

### Alternative G128 FOEM checkpoint

The pinned alternative is
[`XReyRobert/Qwen3.8-27B-GPTQ-Pro-FOEM-4bit-g128-ns256`](https://huggingface.co/XReyRobert/Qwen3.8-27B-GPTQ-Pro-FOEM-4bit-g128-ns256),
revision `af0974575370a06e2feea9926f63f946b6d551a0`. All downloaded LFS
SHA256 hashes and file sizes passed verification. Its tokenizer vocabulary,
complete source encodings and all frozen panel token IDs match the original.
The indexed weights total 19,559,449,368 bytes, versus 19,559,450,216 for the
current checkpoint; both native loader logs report 16.59 GiB model memory.
This does not validate the 200K production profile or MTP execution.

The alternative ran in the same production image and with the same native
capture settings as the existing GPTQ arm: FP16 activations/KV, no W4A8,
no MTP, eager execution, identical 16 windows. Original and EXL3 captures
were reused, not recomputed.

| Checkpoint | Code PPL change vs BF16 | Code mean KL | Prose PPL change vs BF16 | Prose mean KL |
| --- | ---: | ---: | ---: | ---: |
| Current GPTQ G128 | +9.3208% | 0.106451 | +1.7672% | 0.066288 |
| XReyRobert G128 FOEM | +6.0747% | 0.081762 | +4.7941% | 0.061207 |
| EXL3 4 bpw / 6-bpw head | +1.6581% | 0.051717 | +0.4237% | 0.011269 |

FOEM reduces overall mean KL by 17.23%, but overall PPL is effectively tied
with the current checkpoint: paired NLL difference -0.000416 nats/token,
95% window-bootstrap interval [-0.01952, 0.01826]. Its code PPL is 2.97%
lower than current GPTQ, while prose PPL is 2.97% higher. EXL3 remains closer
to the original on both subsets and overall. FOEM-minus-EXL3 paired NLL is
0.042564 nats/token, interval [0.03418, 0.05049]. These exploratory intervals
use 10,000 paired window draws with Python random seed 20261001; the original
comparison intervals use NumPy draws with the same seed.

The follow-up did not establish a broad quality improvement sufficient to
replace production. The original production service/image was restored and
verified healthy. Full metrics are in `comparison-foem.json`, paired
comparisons in `foem-assessment.json`, checkpoint integrity/tokenizer checks
in `gptq-foem-checkpoint.json`, and run/restoration provenance in
`campaign-gptq-foem.json`. The initial `comparison.json` remains unchanged.

The BF16 reference forward took 80.58 seconds and peaked at 5.54 GiB process
RSS and 2.75 GiB allocated XPU memory. The FP16 control took 45.02 seconds.
The native GPTQ and EXL3 passes took 39.76 and 60.83 seconds, respectively,
including their own initialization; these timings are not engine-speed scores.
All native target NLLs passed the independent prompt-logprobs API check. The
production image was restored and its health endpoint returned HTTP 200.

Machine-readable scores are in local `comparison.json`, per-window statistics
and complete FP32 distributions remain in the arm directories, and
`completed.json` records final provenance and the production restoration.

## Frozen inputs

- Original: `Qwen/Qwen3.8-27B`, revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, complete 55.586-GB snapshot.
- GPTQ: `mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, revision
  `a47b0c6f0d756bc394c4cc629d5b0ded1acc7001`.
- EXL3: `turboderp/Qwen3.8-27B-exl3`, 4.00-bpw revision
  `113cf7ab958054860e43fb7f3063b1af19171095`; its output head uses 6 bpw.
- Panel seed: `20261001`; 16 independent 1,024-token windows. Eight windows
  span WikiText-2 raw test; four use CPython 3.12.10 standard-library source;
  four use Vue 3.5.13 reactivity TypeScript source.
- Token-ID vocabularies and source encodings are checked across all three
  tokenizers. No chat template or generated continuation is involved.
- Sources, hashes, token offsets, IDs and sampled prediction positions are
  frozen in `panel.json`. Its SHA256 for the initial run is
  `cbf1a71bbda470859f2c0786cb7134e260111d2072c4753a6732021dadae6171`.

The panel and outputs live in the ignored
`benchmark-results/quantization-reference-20261001` directory. The full-vocabulary
reference distributions alone occupy approximately 509 MB in FP32.
`bundle_reference.py --root /results` creates the portable
`reference-bf16.npz` containing the exact FP32 distributions, all target NLLs,
token IDs, positions and provenance. The initial bundle is 231,118,888 bytes;
compression is lossless and an exact round-trip check is required. Later
checkpoint comparisons can reuse this file without running the original again.

## Measurements

Perplexity is `exp(mean negative log-likelihood)` over all 16,368 next-token
predictions. Report prose and code separately; the combined number is a mixed
panel score, not a standard full-corpus WikiText result.

KL uses 32 uniformly sampled positions per window, without replacement,
from input positions 128 through 1,022: 512 positions in total. Position `j`
predicts token `ids[j+1]`. Each saved distribution covers all 248,320 logits,
including padded vocabulary entries. There is no top-k truncation or floored
tail. Compare `KL(original || quantized)` in nats; also retain reverse KL,
Jensen-Shannon divergence, top-1 agreement and per-window results. Normalize
saved FP32 log-probabilities in FP64 before calculating divergences.

The original runs through the installed Transformers implementation, loading
one decoder layer at a time, applying it to every window, then releasing it.
Embeddings, final norm and output head are streamed separately. Vision and MTP
are downloaded as part of the complete checkpoint but are not executed for
this text likelihood test. `validate_stream.py` checks both attention types
against an ordinary Transformers forward on a small randomized hybrid model.

A second unquantized FP16 pass measures numerical deviation from BF16. Native
quantized passes use their existing vLLM images, FP16 activations/KV, eager
execution, no prefix reuse and no speculative decoding. GPTQ W4A8 and EXL3
INT8 prefill are initially disabled to measure the weight checkpoints without
additional activation quantization. The existing production attention policy
stays enabled; its oneDNN long-context route is ineligible at 1,024 tokens.

Native logits are captured in prompt scoring before sampling. The public
prompt-logprobs API independently verifies all captured target likelihoods.
Incomplete, repeated or misaligned chunks cause the run to fail. Runtime
capture supports both the current XPU V2 prompt worker and the older V1
prompt scorer; `validate_capture.py` tests split-prefill alignment and exclusion
of the final input logit, whose following token is outside the panel.
Triton caches are isolated by arm because the two images use different SYCL
library versions. The older EXL3 runtime needs a bridge interface for oneCCL's
single-rank initialization; the test container publishes no ports.
Runtime
implementations differ, so native results include kernel numerical effects;
this diagnostic does not attribute every difference solely to weight packing.
Window-bootstrap intervals describe uncertainty on this small panel, not
general coding ability or long-context behavior.

## Running

Use `build_panel.py` inside the production image with the three cached snapshot
paths. Keep the resulting panel fixed for later checkpoints.

From the repository root, after creating the panel:

```sh
python benchmarks/experiments/quantization-reference/validate_stream.py
python benchmarks/experiments/quantization-reference/run_campaign.py
```

The validation requires the image's Torch/Transformers environment. The host
campaign runner uses Docker and temporarily stops `b70-qwen38-vllm.service`,
then leaves GPTQ offline by default during EXL3 development. Use the explicit
`--restore-production` flag only for a requested rollback drill. It limits each
test container to 12 GiB RAM and mounts model caches read-only. It refuses to
overwrite completed measurement arms. `--arms` and `--manifest-name` allow a
failed native arm to resume without recomputing a completed reference.

For a cached alternative checkpoint, choose a separate result label and a
pinned HF snapshot. The completed follow-up used:

```sh
python benchmarks/experiments/quantization-reference/run_campaign.py \
  --arms gptq-foem --gptq-arm gptq-foem \
  --gptq-snapshot hub/models--XReyRobert--Qwen3.8-27B-GPTQ-Pro-FOEM-4bit-g128-ns256/snapshots/af0974575370a06e2feea9926f63f946b6d551a0 \
  --manifest-name campaign-gptq-foem.json
```

`download_checkpoint.py` can fetch a pinned snapshot in the image's Python
environment and verify its LFS hashes and tokenization against the frozen
panel before running the GPU arm. It requires `--repo`, `--revision`,
`--root`, `--label` and `--reference-tokenizer`.

Run `compare.py --root /results` inside the production image with the result
directory mounted there. Add native activation-quantization diagnostics using
`--arms gptq-w4a8 exl3-int8`; include these names in `compare.py --arms`.
For the FOEM comparison, use `--arms fp16 gptq gptq-foem exl3
--output-name comparison-foem.json` to preserve the initial result file.

Input sources: [original model](https://huggingface.co/Qwen/Qwen3.8-27B),
[WikiText](https://huggingface.co/datasets/Salesforce/wikitext),
[CPython](https://github.com/python/cpython/tree/v3.12.10/Lib),
[Vue](https://github.com/vuejs/core/tree/v3.5.13/packages/reactivity/src).
Downloaded sources retain their upstream licenses; raw source copies remain
in the local result directory.

## Target-runtime staged quality

The migration runner reuses this exact BF16 panel and verifies all local
reference arrays against the immutable portable bundle. It requires a
completed full equal-contract qualification of the exact candidate image
and serializes work behind the common GPU lock. GPTQ stays offline.

```sh
python3 scripts/run-exl3-quality-stages.py \
  --image sha256:046ef4c7937d0e6f8945191671d7147270f7e24f1772159c48b395ad47451088 \
  --contract-campaign benchmarks/results/exl3-migration/target-contract-weakref-v4 \
  --output benchmark-results/exl3-quality-target-stages-FRESH
```

Five separate arms add FP8 KV, INT8 prefill, graph configuration and MTP3
one at a time to the new FP16 port. These short-panel arms use the
1024-token scoring workload, not the full serving capacity/media profile.
`--reuse-completed PREVIOUS_CAMPAIGN` preserves completed matching arms
when a later arm failed; completed arrays are reused by symlink with image,
config and summary-hash checks. The pruned draft vocabulary is an image
model-profile sidecar, not a Hugging Face checkpoint file.

The comparison keeps all 248320 logit rows, reports mass outside the frozen
tokenizer's 248077 IDs and applies no additional masking. It also reports
paired NLL changes versus the old EXL3 runtime. Configuring graph capture
and MTP is not proof of generated decode/acceptance correctness: generated
probes, long-context suffix/reference checks and the full candidate serving
profile remain separate release gates. Raw arrays stay under the ignored
`benchmark-results` directory; publish lean provenance/metric JSON only.

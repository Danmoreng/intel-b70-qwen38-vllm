# M06 INT4 target-head offline gate

This arm compares the unchanged FP16 target head with INT4/G128 on captured
real target hidden states. It measures operator speed, top-1 agreement,
top-20 recall, logit error and KL divergence. Unlike draft-head quantization,
this changes the target distribution; a positive speed screen is not enough
for serving promotion. NLL and representative task qualification are mandatory.

## Serving result, 19 September 2026

The repaired candidate serves the pinned MTP4/vision profile. The synthetic
decode screen measured +12.55% at 8K, +5.08% at 64K and +16.00% at 128K;
prefill was effectively unchanged. There are only two measurements per arm at
8K/64K and one at 128K, with visible MTP-acceptance variation. Production remains
on M04 with its FP16 target head. See [full results and quality limits](SERVING-2026-09-19.md)
and [machine-readable evidence](serving-results-20260919.json).

## Real-agent result, 19 September 2026

The paired coding run at 180 W favors the existing M04 profile for completed
work: M04 finished both tasks in 42:47 and passed 9/9 independent acceptance
checks. The target-head candidate reached the common 100,000-output-token
budget after 53:49, with its second task unfinished and 8/9 checks passing.
Aggregate native decode was 54.39 versus 55.82 tok/s, but context distributions
and generated content differed substantially. Prefill was close within common
context bands; decode gains varied with MTP acceptance. No promotion follows
from this single pair. See the [full task/context/energy comparison](CODING-RESULTS-2026-09-19.md)
and [numeric results](coding-results-20260919.json).

## Original offline screen

The clean V2-runner capture yielded 36 distinct real target states. INT4
matched FP16 top-1 for all 36, retained 96.11% of the FP16 top-20 set, and had
mean KL(FP16 || INT4) 0.00463 (maximum 0.02716). At 36 rows, full projection
plus argmax improved from 4.868 to 1.468 ms (+231.7%). This justifies a broader
quality arm, not production use. The earlier static-buffer captures are invalid
for quality conclusions and are intentionally not summarized as evidence.

The first serving image retained both FP16 and INT4 target-head weights and
failed the 200704-token startup gate (estimated maximum 194688). Releasing the
source inside the language model's `load_weights` then failed because the outer
loader visits that module more than once as checkpoint prefixes alternate.

The September 19 repair packs only after the outer multimodal loader exhausts
the full checkpoint. It also fixes a second dependency: the executing **V2**
`load_eagle_model` shares the target `lm_head` with Qwen MTP. The older Step3.5
proposer's separate-head policy does not describe this runner. The packed head
therefore travels on the shared module, and the draft builder adopts the same
INT4 tensors before accessing the freed FP16 source. Target embeddings stay
separate (`tie_word_embeddings=false`). TP/PP other than one, tied embeddings,
missing head weights and subsequent weight reloads are refused.

The candidate is scoped to the pinned multimodal Qwen/MTP4 serving profile.
Standalone text-model loading, LoRA and online weight reload are not qualified.
The production M04 image and its service remain the control.

Build and run the isolated screen:

```bash
docker build -f benchmarks/experiments/m06-target-head-int4/candidate/Dockerfile \
  -t local/qwen38-b70-vllm:m06-target-int4-fixed-20260919 .
docker run --rm -v "$PWD:/work:ro" --entrypoint python \
  local/qwen38-b70-vllm:m06-target-int4-fixed-20260919 \
  /work/tests/unit/test_target_head_loading.py
./scripts/start-target-head-int4-serving.sh
```

The runner pins the candidate image ID, checks production is idle, takes the
existing experiment lock, and restores the original production service on exit.
Its candidate/control/control/candidate order measures 8K and 64K twice per arm,
and 128K once per arm, after a full-length warmup. Prefix-cache hits and truncated
benchmark outputs are rejected. Coding assertions, short teacher-forced NLL,
vision/tool calls and prefix-state reuse are checked separately. This is a
screen, not automatic promotion or a broad quality qualification.

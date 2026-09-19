# DFlash2 against the recorded production coding benchmark

Status, 2026-09-19: **rejected for production at the user's required 196 Ki-token
context (200,704 tokens)**. No further DFlash2 optimization or long run is planned.
The prepared workload and evidence are retained as an archive.
No DFlash coding throughput or quality result exists from this study.
The existing production profile is not changed or promoted by these scripts.

## Reused reference

The reference is the completed control arm of
`../m06-target-head-int4/runs/coding-20260919-101848/`:

- Source revision `3ea7cc6cd0ff9422d3bffe54a94aa3cc14506d5c`.
- Two linked tasks: preserve configuration during credential rotation, then
  distinguish disabled/disconnected/ready runner states.
- Pi 0.85.1, medium thinking, identical initial task and follow-up, frozen
  dependencies and independent nine-check acceptance suite.
- 60 minutes, 160 requests, 100,000 total output tokens, 16,384 per request.
- Recorded result: both tasks completed in 2,567.412 seconds, 133 requests,
  maximum actual prompt 143,034 tokens, 9/9 acceptance checks.
- Reference image `sha256:aee9857bef1f37c8f0ee136d9f89d7166201212175a8b171d958627706cf1c0b`.
  Q128 + M04, MTP4, full vocabulary, FP16 target activations, GPTQ INT4/G128,
  FP8 KV, 200,704-token ceiling, APC align, batch 4096, one sequence, 180 W.

The prepared copy is recorded in `LATEST`. Source, harness, dependencies and
draft bytes are checked with hashes. Each coding invocation creates its own
workspace; it never uses today's dashboard with the fixes already present.
Prompts, source, tool output and raw replies remain local in ignored run folders.

Only the candidate needs a new coding run. The saved M04 result is reused, not
silently rerun or retuned. This is a historical-reference comparison of adaptive
agent trajectories, not a controlled replay of identical generated answers.
Whole-session throughput differences cannot alone establish an engine speedup.

## Candidate and prerequisite findings

Same pinned production image and GPTQ target revision
`a47b0c6f0d756bc394c4cc629d5b0ded1acc7001`. The candidate replaces MTP with:

```json
{"method":"dflash","model":"/draft","num_speculative_tokens":7}
```

Draft: `incoai/Qwen3.8-27B-DFlash2`, revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`; safetensors SHA-256
`67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.
It already exists locally under `models-legacy`; no replacement was downloaded.

Configuration-only checks accepted FP16 and BF16, selected the V2 runner and
retained APC. This does **not** establish inference correctness. The actual
candidate uses **BF16 target and draft activations**, because the stock config
inherits target dtype for the draft. The known FP16 DFlash2 overflow problem
has an upstream fast-fail proposal, not a mixed-dtype implementation:

- https://github.com/vllm-project/vllm/issues/55250
- https://github.com/vllm-project/vllm/pull/55294

This changes more than the speculation method. In particular, the installed
Q128/M04 attention adapter requires `q.dtype == torch.float16`; BF16 falls back
to native attention. A future mixed FP16-target/BF16-draft implementation would
need separate numerical, shared-head/embedding and cache validation. The current
candidate is not advertised as retaining those fast paths.

## Actual startup result

Probe: `runs/prepared-20260919-164436/probe-20260919-164705/engine.log`.

| Observation | Value |
|---|---:|
| Target + draft model allocation | 21.2 GiB |
| Available KV cache after profiling | 3.32 GiB |
| Required KV cache at 200,704 tokens | 9.24 GiB |
| Engine-estimated maximum context | 44,928 tokens |
| Longest prompt in the reference coding run | 143,034 tokens |

The engine failed its capacity check before serving any request. Acceptance,
prefix-state correctness and coding quality are therefore **unmeasured**.
The 44,928 value is a startup estimate for this candidate, not a qualified
working context or a permanent DFlash architecture limit.

No context reduction or early compaction is substituted into the comparison.
Reducing the ceiling to 32K/44K would change the workload and cannot repeat the
reference's context trajectory. Increasing the memory fraction alone cannot
cover the approximately 5.9 GiB deficit at the original ceiling.

The user rejected this candidate because the production context is a hard
requirement. Draft quantization, memory optimizations, reduced context and a
mixed-dtype port are not being pursued. These scripts are archived preparation,
not an instruction to resume the experiment.

### What the approximately 6 GiB deficit actually means

The reference's `arm-0-control/engine.log` reports 17.69 GiB at model loading and
8.02 GiB available KV memory. DFlash reports 21.20 and 3.32 GiB respectively:

| Startup accounting | MTP4 reference | DFlash2 candidate | Difference |
|---|---:|---:|---:|
| Model-load allocation | 17.69 GiB | 21.20 GiB | +3.51 GiB |
| Available KV budget after profiling | 8.02 GiB | 3.32 GiB | -4.70 GiB |

Approximately 1.19 GiB of the KV-budget loss is therefore outside the reported
model-load difference. This is a residual from the engine's startup accounting;
the available logs do not attribute it precisely to individual temporary
activations, persistent buffers or non-PyTorch allocations. It must not be
labelled as a measured graph allocation: XPU graph estimation is excluded in
this worker's pre-KV profiling path.

DFlash uses a separate five-layer draft checkpoint, about 1.92 billion
parameters, stored in BF16 (3.58 GiB checkpoint), plus its cache/state machinery.
The current MTP draft is much smaller and its heavy linear weights are INT4.
The target GPTQ weights stay INT4. FP16 and BF16 both use two bytes per value;
the activation dtype switch by itself does not double storage.

9.24 - 3.32 = **5.92 GiB** is the missing KV budget at the requested context,
not the draft checkpoint size or a measurement of six additional GiB of weights.
DFlash's cache grouping also logs padding overhead; the memory-to-context mapping
includes fixed hybrid-model state costs and is not a simple proportional scale.
The 44,928-token estimate is about 22% of the required 200,704-token ceiling.

## Prepared execution

From this directory:

```bash
python3 prepare.py                 # make a new frozen candidate-only study
bash start.sh preflight            # exclusive, short serving qualification
bash start.sh coding               # refuses unless all gates passed
```

`start.sh` uses systemd with recovery on normal exit, failure and cancellation.
The shared GPU experiment lock and idle check run before stopping production.
Recovery checks the restored image, 180 W power cap and HTTP health endpoint.
Stop an active experiment with `systemctl --user stop b70-m08-dflash2.service`;
the recovery action still runs. Status is stored in the run's `state.json` and
`preflight.json`; `production-restored.json` records successful restoration.

Before allowing coding, the harness requires:

1. Actual startup with the reference's 200,704-token ceiling.
2. Nonzero accepted draft tokens.
3. Correct cold/warm prefix-state probes.
4. Successful inference on the saved 143,034-token reference prompt.
5. An isolated agent read-tool/follow-up smoke with valid native accounting.

The last four gates have not run because startup failed. `preflight.json` stays
`ready: false`; no long coding run was started. A profile change invalidates an
earlier passing gate through the manifest hash.

The long-run harness reuses the reference's measurement proxy and evaluator:
native prefill/decode counters per request, weighted 10K context bands, cache
hits, speculation acceptance, task and tool time, compaction and card energy.
It produces `comparison.json` with the recorded control and new candidate.
Output tokens include reasoning; unvisited bands remain unmeasured. Format
warnings are compared with the frozen baseline, separately from functional
checks. The agent/evaluation branch has not been exercised end to end with
DFlash because of the serving blocker.

Validation of the new supervisor: four unit tests cover preserved configuration,
manifest-bound qualification, ownership checks and recovery failure handling;
Python compilation and shell syntax checks pass. The actual failed startup
exercised the production restoration path.

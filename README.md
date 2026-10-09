# Intel Arc Pro B70: Qwen3.8-27B with vLLM

The qualified production profile serves **EXL3 4.00 bpw on one 32 GB Intel Arc
Pro B70 at 180 W**, using vLLM 0.31.0 and Torch 2.14.0+xpu. It exposes an
OpenAI-compatible API at `http://127.0.0.1:8081/v1`, model `Qwen3.8-27B`.

## Start and recover the qualified service

The local release uses the immutable image recorded in
[config/production_image.json](config/production_image.json):
`sha256:194a8786f12065e65647d914ddc7fdc136bafbeec73f476730ef1324e4d2c7ec`,
alias `local/b70-qwen38-vllm:production-exl3-v3`.
The strict launcher checks image/policy identity, native libraries, serving
arguments, checkpoint files, middleware and the 180 W cap before loading.
Compiler caches are isolated by policy and image ID.

On an installation with the qualified image and local checkpoint already
available, run from the checkout used by the user service:

```bash
cp .env.example .env                 # initial setup only; retain an existing .env
./scripts/download-model.sh
./scripts/set-power-limit.py
./scripts/install-user-service.sh
curl -fsS http://127.0.0.1:8081/v1/models
```

Check or recover an existing installation:

```bash
python3 scripts/run-server-exl3.py --check-only
python3 scripts/restore-production.py
systemctl --user status --no-pager b70-qwen38-vllm.service
journalctl --user -u b70-qwen38-vllm.service -n 60 --no-pager
```

The restore helper validates current release receipts, the service checkout
and launcher preflight before starting the service. The service uses Type=exec;
its process being active does not mean model loading has finished. Check the
API readiness endpoint. The host defaults are loopback and port 8081; the
recovery helper targets that endpoint and the standard container name.

The immediate rollback is the [saved EXL3 v2 release](config/releases/exl3-v2/production_image.json),
with its [strict launcher](scripts/run-server-exl3-v2-rollback.sh), policy, source
snapshot and compiler namespace. The older
[saved EXL3 v1 rollback](config/releases/exl3-v1/README.md) also remains available.

The [EXL3 source snapshot/build instructions](engine/exl3xpu/README.md) contain
immutable upstream and native/header pins. A rebuilt image needs its own
qualification and release manifest. For a **base development build**, use a
separate tag:

```bash
B70_BASE_BUILD_IMAGE=local/b70-qwen38-vllm:base-development ./scripts/build-image.sh
```

This helper does not load the serving `.env` and rejects production/rollback
aliases before any build or download. It builds the base image; use the EXL3
build workflow to produce an EXL3 candidate.

## API defaults and example

Chat requests default to 16,384 maximum completion tokens and an 8,192-token
thinking budget. Larger thinking budgets are capped at 8,192. Reasoning effort
is `medium`; thinking and retention of thinking history are enabled by default.
`reasoning_effort: "none"` disables thinking unless explicitly overridden with
`chat_template_kwargs.enable_thinking`. Requests can set a smaller output cap.
Tools use the `qwen3_xml` parser and automatic tool choice.

```bash
curl -fsS http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B","messages":[{"role":"user","content":"Write a small JavaScript debounce function."}],"max_tokens":512,"reasoning_effort":"none"}'
```

## Current production configuration

| Setting | Applied value |
|---|---|
| Model / revision | `turboderp/Qwen3.8-27B-exl3` / `113cf7ab958054860e43fb7f3063b1af19171095` |
| Served name | `Qwen3.8-27B` |
| Weights / activations / target head | EXL3 4.00-bpw checkpoint / FP16 / 6-bpw full 248,320-row head |
| Linear dispatch | Native EXL3 SmallM through 128 rows; INT8 large-matrix prefill; unchanged row policy |
| Prefill attention | Guarded oneDNN: query rows ≥64, exact active KV 4,096–262,144, query bucket 256, exact-K bucket 1; eager only |
| Verification | M04 for supported uniform q2–5 / C1–C4; direct output copy at C4; native fallback elsewhere; no duplicate KV updates |
| oneDNN partition cache | 64 exact-shape entries per inference thread/queue; completion-aware LRU eviction; oneDNN internal caches are a separate scope |
| MTP | 3 draft tokens, FP16; 65,536-row draft vocabulary; full 248,320-row target vocabulary |
| Context | 262,144 total input+output tokens |
| Admission / batch | 16 sequences; 4,096 max batched tokens; full-ISL, watermark 0.0 |
| Graphs / cache | FULL_DECODE_ONLY, capture sizes 1/2/4/8/12/16/24/32/40/48/56/64; FP8 KV, prefix reuse, Mamba alignment |
| Memory / power | 0.965 GPU fraction / 180 W card cap |
| Runtime | vLLM 0.31.0, Torch 2.14.0+xpu, oneAPI 2026.1.1 / SYCL 9, oneDNN 3.13.0; Intel Runtime 26.35.39758.10, IGC 2.41.5 |
| Tools / reasoning / media | qwen3_xml / qwen3; 32 images or 4 videos within the total context limit; image cap 4,194,304 pixels |

## Capacity and operating limits

- **262,144 total tokens** includes input, retained conversation/reasoning,
  media tokens and generated output. The exact **261,120 input + 1,024 output**
  boundary is qualified.
- **C4 at moderate context** is qualified without preemptions. Sixteen admitted
  requests share the same cache pool. All 16 pressure-test requests completed,
  with 2 preemptions; this allows waiting/recomputation and
  does not promise sixteen simultaneous maximum contexts.
- Up to **32 images or 4 videos** per prompt, within the same context and
  memory budget. The image cap is 4,194,304 pixels. Qualification includes
  32 distinct images, long image context, video limits and recovery.
- Independent FP32 kernel comparisons and restored-state prefill, mixed-batch,
  C4-graph and long-prefill numerical probes pass at the original tolerances.
  Output distributions and generated text change across Torch/vLLM versions.
  The old same-runtime bitidentity check failed and remains recorded; the
  cross-version changes were reviewed explicitly. Finite probes do not
  guarantee identical text or all-task accuracy.
- Historical EXL3 v1 QueueKit passed **7/8 checks and 1/2 tasks**, with the
  same failure in its native-attention control. Coding tasks were not rerun
  for this release; the failed task remains documented.

See the [current v3 qualification and limitations](benchmarks/results/vllm-031-upgrade/assessment.json),
[stored-baseline comparison](benchmarks/results/vllm-031-upgrade/benchmark-comparison.json)
and [reviewed performance tradeoff](benchmarks/results/vllm-031-upgrade/performance-review.json).
The 64K C1 decode median fell **52.67 → 48.29 tok/s (−8.3%)**, reproduced on
a fresh new-image worker. MTP acceptance fell **60.7% → 56.1%**, while measured
round-equivalent cost stayed close; the exact upstream cause is unproven.
The other 19 load points have no native-decode decrease or TTFT increase
crossing the 5% review trigger. Performance changes are mixed.

The [v2 report](docs/EXL3_RELEASE_V2_REPORT.md),
[v1 report](docs/EXL3_RELEASE_REPORT.md), coding measurements and
[bounded review experiments](docs/EXL3_PRO_REVIEW_FOLLOWUP.md) remain historical.
The serving measurements below are fresh measurements of the deployed v3 image.
[Runtime distribution and ABI pins](config/experiments/vllm-031-upgrade/runtime_requirements.json)
include the official XPU image’s inherited dependency metadata conflicts;
actual XPU/native and serving qualification applies to this exact pinned image.

<!-- BEGIN CURRENT SERVING MEASUREMENTS -->
## Source-review serving benchmark

Measured **2026-10-09** on a fresh isolated worker with the
[frozen public corpus](benchmarks/meaningful-corpus.json). The
[current summary](benchmarks/results/vllm-031-upgrade/serving-summary.json) records scenario results, fixture hashes,
fixed prompt namespace `20260923-201101` and image identity. Raw prompts,
responses and stream events remain local under `benchmark-results/`.

The complete run covered **20 scenarios, 70 measured waves and 124 successful
requests** in **53.9 minutes**. Each request sampled at temperature 1.0,
top-p 0.95 and top-k 20 with thinking disabled. `ignore_eos=true` required
exactly 1,024 output tokens. These are throughput measurements rather than
semantic answer scores. All prompt/output counts matched, all requests
finished at the output cap, with **0 preemptions** and **0 excess recomputed
prefill tokens**. All 124 measured prompts were verified byte-for-byte
against the frozen original full-run fixtures; request payload values match.
Three additional 64K resends on a fresh worker on 2026-10-09 supply the
cold/warm 64K row, because the complete run's first 64K prefix request
reused 14,400 tokens from preceding requests. Those extra prompts and
payloads match the same frozen fixtures.

### One request: context sweep

Input values are token budgets. Prefill is newly computed KV tokens per native
prefill second; decode is post-first generated tokens per native decode second.
Rates and latencies are medians across waves. MTP acceptance is weighted
across drafted tokens.

| Input / output budget | Waves | Actual input | Prefill tok/s | Decode tok/s | MTP accepted | TTFT | End to end |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 / 1,024 | 5 | 479–507 | 1,707.4 | 58.4 | 52.7% | 0.30 s | 17.82 s |
| 2,048 / 1,024 | 5 | 1,992–2,047 | 2,492.7 | 59.6 | 59.1% | 0.82 s | 17.98 s |
| 4,096 / 1,024 | 5 | 4,052–4,094 | 2,291.4 | 58.1 | 54.0% | 1.79 s | 19.41 s |
| 8,192 / 1,024 | 5 | 8,167–8,186 | 2,325.5 | 55.0 | 52.3% | 3.53 s | 22.13 s |
| 16,384 / 1,024 | 5 | 16,335–16,379 | 2,183.4 | 56.7 | 55.5% | 7.52 s | 25.55 s |
| 32,768 / 1,024 | 5 | 32,704–32,762 | 1,990.1 | 55.2 | 56.4% | 16.50 s | 35.02 s |
| 65,536 / 1,024 | 3 | 65,491–65,532 | 1,679.0 | 48.3 | 56.1% | 39.09 s | 60.27 s |
| 131,072 / 1,024 | 3 | 131,034–131,070 | 1,267.8 | 42.0 | 55.6% | 103.55 s | 127.96 s |

### One to four simultaneous requests

The rates count aggregate generated tokens in sampled intervals after all
requests emitted a first token and before any finished. C1 comes from the
context sweep; C2–C4 each include three waves with different tasks. These are
separate serving load points, not paired scaling measurements.

| Input / output per request | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| 2,048 / 1,024 | 60.5 tok/s | 105.4 tok/s | 153.0 tok/s | 189.8 tok/s |
| 4,096 / 1,024 | 57.4 tok/s | 107.5 tok/s | 145.5 tok/s | 180.0 tok/s |
| 16,384 / 1,024 | 56.2 tok/s | 101.6 tok/s | 131.4 tok/s | 159.0 tok/s |

Some requests briefly entered the scheduler waiting queue; no request was preempted.

### Prefix reuse and maximum context

| Scenario | Measured result |
|---|---|
| 16K exact resend | 14,400 / 16,382 prompt tokens cached; TTFT **7.54 s cold → 0.99 s warm** |
| 64K exact resend | 62,400 / 65,469 prompt tokens cached; TTFT **38.82 s cold → 2.36–2.38 s warm** |
| Frozen 200K comparison | **199,673 input + 1,024 output**; 1,002.6 prefill tok/s, 37.9 decode tok/s, 199.40 s TTFT, 226.35 s end to end |

The longest-context row is one capacity and throughput observation. The
16K/64K resends reused the same prompt on the same worker.

<!-- END CURRENT SERVING MEASUREMENTS -->

<!-- BEGIN POWER COMPARISON -->
## Historical EXL3 v2 power-limit comparison: 150 / 180 / 230 / 275 W

These measurements belong to the saved EXL3 v2 release, with vLLM 0.30.0
and Torch 2.13.0+xpu. All campaigns used the same immutable image, serving
arguments, environment values, frozen prompts and sampling. The existing fresh 180 W matrix is the baseline; the
other measured variants each repeat **20 scenarios / 70 measured waves / 124 requests** plus
an isolated 64K cold/warm resend. All ordinary requests completed without
preemptions. The production service is restored to **180 W**.

These are measured hardware variants; they do not change the qualified
production policy. Power limits are not assumed actual consumption: the table
uses the card energy counter. Measured wave times exclude warmup/startup;
energy includes each measured wave's prefill, decode and request overhead.

| Power limit | Measured mean card power | Wave time | Output tok/s | Output tok/s per W | Output tok/Wh | Card energy |
|---:|---:|---:|---:|---:|---:|---:|
| 150 W | 150.1 W | 50.05 min | 42.3 | 0.282 | 1,014.5 | 125.16 Wh |
| 180 W | 180.1 W | 41.68 min | 50.8 | 0.282 | 1,015.0 | 125.10 Wh |
| 230 W | 230.1 W | 35.77 min | 59.2 | 0.257 | 925.7 | 137.16 Wh |
| 275 W | 275.0 W | 33.89 min | 62.4 | 0.227 | 817.4 | 155.34 Wh |

All efficiency columns cover the same **126,976 output tokens**, including
their prefill and request overhead. `Output tok/s per W` divides whole-wave
throughput by measured mean card power: it equals output tokens/J. `Output
tok/Wh` is output tokens divided by integrated card energy; **higher is
better**. These are not decode-only rates, and do not include whole-PC power.

For this fixed mixed workload, **180 W produced the most tokens per Wh**;
**275 W finished the measured waves fastest**. The rate tables below show
which context and concurrency points benefit from the additional power.

The observed efficiency lead over the next-best setting (150 W) is
**0.1%**. Each limit has one full campaign with the fixed
repeats, without randomized order. Small differences do not establish a
general optimum; generated trajectories and MTP acceptance can also differ.

C1 rates below are median native prefill / request-weighted decode in tok/s.

| C1 input budget | 150 W prefill / decode | 180 W prefill / decode | 230 W prefill / decode | 275 W prefill / decode |
|---:|---:|---:|---:|---:|
| 4K | 1,858.0 / 45.8 | 2,241.4 / 57.1 | 2,615.2 / 66.8 | 2,791.5 / 69.5 |
| 16K | 1,785.8 / 46.5 | 2,168.2 / 56.2 | 2,540.1 / 65.5 | 2,721.1 / 68.4 |
| 64K | 1,396.6 / 43.6 | 1,670.0 / 52.7 | 1,947.6 / 60.6 | 2,086.1 / 63.9 |
| 128K | 1,069.5 / 33.3 | 1,262.5 / 39.9 | 1,466.3 / 45.9 | 1,562.6 / 48.0 |
| 200K | 852.5 / 32.6 | 999.3 / 38.5 | 1,154.4 / 44.0 | 1,228.2 / 46.2 |

C4 rates are fully overlapped aggregate decode in tok/s, using the same
sampled-interval definition as the main serving table.

| Input per request, C4 | 150 W aggregate decode | 180 W aggregate decode | 230 W aggregate decode | 275 W aggregate decode |
|---|---:|---:|---:|---:|
| 2K | 156.1 | 190.5 | 223.5 | 231.1 |
| 4K | 146.9 | 180.1 | 204.7 | 219.0 |
| 16K | 138.2 | 165.0 | 191.5 | 202.2 |

[All 20 load points, acceptance, energy scope and temperatures](docs/EXL3_POWER_COMPARISON.md); [measurement receipts](benchmarks/results/exl3-power-20261002-with-150w/comparison.json).
<!-- END POWER COMPARISON -->

## Coding and quality results

The historical qualified v1 release completed Flappy Bird v7 in **24 min 12 s**, passed
**54/54 checks**, and reaches **90,444 input tokens**. Native weighted prefill /
decode were **1,388.7 / 53.9 tok/s**. Adaptive histories and tool work contribute
to task duration. QueueKit took **7 min 36 s**, with the failure noted above.
[Full coding results and 10K context bands](docs/EXL3_CODING_BENCHMARKS.md).

The new v3 run of the frozen short panel gives PPL **3.64217** (stored v2:
**3.64672**), with mean KL against the stored BF16 reference **0.033353** (stored v2: **0.032481**).
Thus PPL improves slightly while mean KL increases. The 32K suffix PPL is
**1.19036** versus **1.19419** in stored v2. These finite checkpoint/path
measurements do not establish a general coding ranking or bitidentical text.
[Current quality scope and metric receipts](benchmarks/results/vllm-031-upgrade/assessment.json).

## Rollback and benchmark reproduction

The immediate rollback is the [saved EXL3 v2 release](config/releases/exl3-v2/production_image.json).
Use `scripts/run-server-exl3-v2-rollback.sh`; it retains its own image, policy,
source snapshot and compiler namespace. The saved v1 and GPTQ releases also remain available.
Stop the current service before using that launcher; verify the API and a real reply.

For the additional GPTQ rollback, stop `b70-qwen38-vllm.service` before switching to
`scripts/run-server-gptq-rollback.sh`; both profiles use the same API endpoint
and container name. The [independently pinned GPTQ rollback](config/releases/gptq-onednn-v2/README.md)
uses its own image, policy and compiled caches. For a persistent rollback,
follow that document's user-service instructions and verify a real request.

The [benchmark reproduction guide](docs/EXL3_BENCHMARK_REPRODUCTION.md) preserves
the frozen 20-scenario / 70-wave source matrix, QueueKit and six-stage Flappy
fixture. Run with exclusive GPU access and fresh output directories.
[Historical GPTQ comparisons](docs/EXL3_GPTQ_COMPARISON.md), the
[release report](docs/EXL3_RELEASE_REPORT.md) and
[port history](docs/EXL3_PORT_NOTES.md) retain the migration evidence.

## Sources and acknowledgements

- [Qwen model](https://huggingface.co/Qwen/Qwen3.8-27B)
- [EXL3 checkpoint](https://huggingface.co/turboderp/Qwen3.8-27B-exl3)
- [0xSero EXL3 XPU source](https://github.com/0xSero/exl3xpu)
- [vLLM](https://github.com/vllm-project/vllm)
- [Intel B70 cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)

See [NOTICE.md](NOTICE.md) and the preserved upstream license for attribution.

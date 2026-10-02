# Intel Arc Pro B70: Qwen3.8-27B with vLLM

The qualified production profile serves **EXL3 4.00 bpw on one 32 GB Intel Arc
Pro B70 at 180 W**, using vLLM 0.30.0 and Torch 2.13.0+xpu. It exposes an
OpenAI-compatible API at `http://127.0.0.1:8081/v1`, model `Qwen3.8-27B`.

## Start and recover the qualified service

The local release uses the immutable image recorded in
[config/production_image.json](config/production_image.json):
`sha256:c09015ce22180fbc90ef0f5070f4a7116c8d11be785067499477accf0216f21f`,
alias `local/b70-qwen38-vllm:production-exl3-v1`.
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
| Verification | Rebuilt M04 for supported uniform q2–5 / C1–C4; native fallback elsewhere; no duplicate KV updates |
| MTP | 3 draft tokens; 65,536-row draft vocabulary; full 248,320-row target vocabulary |
| Context | 262,144 total input+output tokens |
| Admission / batch | 16 sequences; 4,096 max batched tokens; full-ISL, watermark 0.0 |
| Graphs / cache | FULL_DECODE_ONLY, capture sizes 1/2/4/8/12/16/24/32/40/48/56/64; FP8 KV, prefix reuse, Mamba alignment |
| Memory / power | 0.965 GPU fraction / 180 W card cap |
| Runtime | vLLM 0.30.0, Torch 2.13.0+xpu, oneAPI 2026.1.1 / SYCL 9, oneDNN 3.13.0; Intel Runtime 26.35.39758.10, IGC 2.41.5 |
| Tools / reasoning / media | qwen3_xml / qwen3; 32 images or 4 videos within the total context limit; image cap 4,194,304 pixels |

## Capacity and operating limits

- **262,144 total tokens** includes input, retained conversation/reasoning,
  media tokens and generated output. The exact **261,120 input + 1,024 output**
  boundary is qualified.
- **C4 at moderate context** is qualified without preemptions. Sixteen admitted
  requests share the same cache pool. C16 pressure tests completed with four
  preemptions affecting two requests; this allows waiting/recomputation and
  does not promise sixteen simultaneous maximum contexts.
- Up to **32 images or 4 videos** per prompt, within the same context and
  memory budget. The image cap is 4,194,304 pixels. Qualification includes
  32 distinct images, long image context, video limits and recovery.
- Generated text can vary with attention route, batching and floating-point
  rounding. The short quality panel does not exercise the ≥4096-token oneDNN
  prefill route or prove identical graph-based MTP verification.
- QueueKit passes **7/8 checks and 1/2 tasks**. The same failed case occurs in
  the native-attention control; the actual task failure remains documented.

See the [qualification and limitations](docs/EXL3_RELEASE_REPORT.md) and the
[bounded Pro-review follow-up](docs/EXL3_PRO_REVIEW_FOLLOWUP.md). Review candidate
measurements are kept separate from the immutable v1 production measurements
below.

<!-- BEGIN CURRENT SERVING MEASUREMENTS -->
## Source-review serving benchmark

Measured **2026-10-02** on a fresh isolated worker with the
[frozen public corpus](benchmarks/meaningful-corpus.json). The
[current summary](benchmarks/runs/2026-10-02-exl3-production/summary.json) records scenario results, fixture hashes,
fixed prompt namespace `20260923-201101` and image identity. Raw prompts,
responses and stream events remain local under `benchmark-results/`.

The complete run covered **20 scenarios, 70 measured waves and 124 successful
requests** in **53.7 minutes**. Each request sampled at temperature 1.0,
top-p 0.95 and top-k 20 with thinking disabled. `ignore_eos=true` required
exactly 1,024 output tokens. These are throughput measurements rather than
semantic answer scores. All prompt/output counts matched, all requests
finished at the output cap, with **0 preemptions** and **0 excess recomputed
prefill tokens**. All 124 measured prompts were verified byte-for-byte
against the frozen original full-run fixtures; request payload values match.
Three additional 64K resends on a fresh worker on 2026-10-02 supply the
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
| 512 / 1,024 | 5 | 479–507 | 1,700.9 | 58.2 | 55.1% | 0.31 s | 17.88 s |
| 2,048 / 1,024 | 5 | 1,992–2,047 | 2,422.1 | 59.7 | 59.5% | 0.84 s | 17.97 s |
| 4,096 / 1,024 | 5 | 4,052–4,094 | 2,235.7 | 55.2 | 51.6% | 1.83 s | 20.38 s |
| 8,192 / 1,024 | 5 | 8,167–8,186 | 2,300.1 | 54.8 | 52.2% | 3.57 s | 22.26 s |
| 16,384 / 1,024 | 5 | 16,335–16,379 | 2,166.2 | 56.1 | 56.3% | 7.58 s | 25.81 s |
| 32,768 / 1,024 | 5 | 32,704–32,762 | 1,974.8 | 56.5 | 60.7% | 16.61 s | 34.70 s |
| 65,536 / 1,024 | 3 | 65,491–65,532 | 1,668.2 | 52.7 | 60.7% | 39.36 s | 58.76 s |
| 131,072 / 1,024 | 3 | 131,034–131,070 | 1,261.8 | 39.9 | 54.5% | 104.05 s | 129.71 s |

### One to four simultaneous requests

The rates count aggregate generated tokens in sampled intervals after all
requests emitted a first token and before any finished. C1 comes from the
context sweep; C2–C4 each include three waves with different tasks. These are
separate serving load points, not paired scaling measurements.

| Input / output per request | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| 2,048 / 1,024 | 60.7 tok/s | 108.5 tok/s | 152.2 tok/s | 185.6 tok/s |
| 4,096 / 1,024 | 55.4 tok/s | 110.9 tok/s | 141.4 tok/s | 181.6 tok/s |
| 16,384 / 1,024 | 56.5 tok/s | 101.5 tok/s | 137.1 tok/s | 168.2 tok/s |

Some requests briefly entered the scheduler waiting queue; no request was preempted.

### Prefix reuse and maximum context

| Scenario | Measured result |
|---|---|
| 16K exact resend | 14,400 / 16,382 prompt tokens cached; TTFT **7.57 s cold → 1.03–1.04 s warm** |
| 64K exact resend | 62,400 / 65,469 prompt tokens cached; TTFT **39.09 s cold → 2.48–2.49 s warm** |
| Frozen 200K comparison | **199,673 input + 1,024 output**; 999.0 prefill tok/s, 38.5 decode tok/s, 200.14 s TTFT, 226.68 s end to end |

The longest-context row is one capacity and throughput observation. The
16K/64K resends reused the same prompt on the same worker.

<!-- END CURRENT SERVING MEASUREMENTS -->

## Coding and quality results

The qualified v1 release completes Flappy Bird v7 in **24 min 12 s**, passes
**54/54 checks**, and reaches **90,444 input tokens**. Native weighted prefill /
decode are **1,388.7 / 53.9 tok/s**. Adaptive histories and tool work contribute
to task duration. QueueKit takes **7 min 36 s**, with the failure noted above.
[Full coding results and 10K context bands](docs/EXL3_CODING_BENCHMARKS.md).

On the frozen short panel, EXL3 PPL is **3.64672**, with mean KL **0.032481**
against BF16. These finite checkpoint/path measurements do not establish a
general coding ranking. [Quality scope and metric receipts](benchmarks/results/exl3-migration/optimized-quality-v1/README.md).

## Rollback and benchmark reproduction

Stop `b70-qwen38-vllm.service` before switching to
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

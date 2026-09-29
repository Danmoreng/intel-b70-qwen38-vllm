# Intel Arc Pro B70: Qwen3.8-27B with vLLM

A single-GPU GPTQ serving recipe for one **32 GB Intel Arc Pro B70 at 180 W**.
The current engine combines W4A8 large-matrix prefill, bounded oneDNN
attention prefill, Q128/native fallback, and M04 shared-KV MTP verification.
It serves one local coding user with up to four active sequences; requests
queue when the KV capacity cannot admit them together.

## Current production configuration

| Setting | Applied value |
|---|---|
| Model | `mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` at revision `a47b0c6f0d756bc394c4cc629d5b0ded1acc7001` |
| Served name | `Qwen3.8-27B` |
| Body / activations / target head | GPTQ INT4 symmetric G128 / FP16 / FP16 |
| Large-matrix prefill | W4A8 from **512 total rows**; existing W4A16 below that threshold |
| Attention prefill | oneDNN for eligible query rows **256–6,656** and exact active KV **16,384–196,608** tokens; Q128/native outside that domain |
| Decode / verification | M04 shared-KV or native path by fixed tensor contract |
| Speculation | MTP4, full 248,320-row draft vocabulary, INT4 draft head and five INT4 MTP linears |
| KV cache | FP8; automatic prefix caching; Mamba cache alignment |
| Context | Up to **200,704** input + output tokens |
| Scheduler | Up to **4** active sequences; 6,656 max batched tokens; full-ISL admission; watermark 0.0 |
| Memory / power | GPU memory fraction 0.93; card power cap **180 W** |
| Runtime | vLLM `0.30.0+xpu`, XPU kernels `0.1.15.4`, Intel Compute Runtime `26.35.39758.10`, IGC `2.41.5` |
| API configuration | `qwen3_xml` tool parser, `qwen3` reasoning parser; at most one image and no video per prompt |

The [frozen policy](config/production_policy.json) has SHA-256
`4ce5f3bd77710fe08ac4b96ee761c50eb13c2d3b48ac74f20ab7fced82b4b368`.
The measured image is
`sha256:a42cda993bf6492acc39d23e9382e27a17efca4657bea07d80d0c28228a1623a`
(`local/b70-qwen38-vllm:production-onednn-v1`). The launcher checks the policy
and image label before serving; the image checks its required native libraries
and source files at startup. Both benchmarks below used this same image and
policy. The source-review run used the permanent `b70-qwen38-vllm.service`.

## Source-review serving benchmark

Measured **2026-09-29** on the permanent service with the
[frozen public corpus](benchmarks/meaningful-corpus.json). The complete
[current summary](benchmarks/runs/2026-09-29-production/summary.json) records
scenario results, fixture hash and image identity. The raw results, prompts,
responses and stream events remain local under `benchmark-results/`.

The fixed run covered **20 scenarios, 70 measured waves and 124 successful
requests** in **63.3 minutes**. Each request sampled at temperature 1.0,
top-p 0.95 and top-k 20 with thinking disabled. `ignore_eos=true` forced
exactly 1,024 output tokens, so these are throughput measurements rather than
semantic answer scores. All prompt and output counts matched, all requests
finished at the output cap, and there were **0 preemptions** and **0 excess
recomputed prefill tokens**.

### One request: context sweep

Input values are token budgets. Actual source-review prompts were close to
those budgets. Prefill is newly computed KV tokens per native prefill second;
decode is post-first generated tokens per native decode second. Rates and
latencies are medians across waves. MTP acceptance is weighted across drafted
tokens.

| Input / output budget | Waves | Actual input | Prefill tok/s | Decode tok/s | MTP accepted | TTFT | End to end |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 / 1,024 | 5 | 494–510 | 2,280.8 | 67.8 | 50.0% | 0.23 s | 15.32 s |
| 2,048 / 1,024 | 5 | 2,022–2,047 | 2,306.1 | 62.6 | 46.0% | 0.89 s | 17.23 s |
| 4,096 / 1,024 | 5 | 4,037–4,086 | 2,146.2 | 62.4 | 46.8% | 1.91 s | 18.30 s |
| 8,192 / 1,024 | 5 | 8,178–8,191 | 2,037.8 | 63.0 | 46.5% | 4.03 s | 20.27 s |
| 16,384 / 1,024 | 5 | 16,362–16,384 | 1,843.6 | 56.7 | 43.9% | 8.91 s | 26.63 s |
| 32,768 / 1,024 | 5 | 32,717–32,765 | 1,808.5 | 58.1 | 50.4% | 18.16 s | 35.77 s |
| 65,536 / 1,024 | 3 | 65,475–65,529 | 1,579.7 | 47.7 | 44.4% | 41.57 s | 63.01 s |
| 131,072 / 1,024 | 3 | 131,018–131,068 | 1,234.6 | 40.1 | 45.0% | 106.31 s | 131.88 s |

### One to four simultaneous requests

The rates below count aggregate generated tokens in sampled intervals after
all requests emitted a first token and before any finished. C1 values come
from the context sweep; C2–C4 each include three waves with different tasks.
These are separate serving load points, not paired scaling measurements.

| Input / output per request | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| 2,048 / 1,024 | 61.7 tok/s | 100.8 tok/s | 142.4 tok/s | 181.0 tok/s |
| 4,096 / 1,024 | 61.9 tok/s | 99.5 tok/s | 139.2 tok/s | 179.4 tok/s |
| 16,384 / 1,024 | 56.6 tok/s | 88.0 tok/s | 123.0 tok/s | 139.2 tok/s |

At C4 the scheduler queued some work when capacity was tight. It did not
preempt any request in this run.

### Prefix reuse and maximum context

| Scenario | Measured result |
|---|---|
| 16K exact resend | 13,312 / 16,328 prompt tokens cached; TTFT **8.89 s cold → 1.89 s warm** |
| 64K exact resend | 63,232 / 65,475 prompt tokens cached; TTFT **41.56 s cold → 2.00 s warm** |
| Maximum context | **199,678 input + 1,024 output**; 981.5 prefill tok/s, 31.0 decode tok/s, 203.70 s TTFT, 236.64 s end to end |

The maximum-context row is one capacity and throughput observation. The
16K/64K resends reused the same prompt on the same worker.

## Repeatable coding-agent benchmark

The [QueueKit fixture v2](benchmarks/coding-fixture/v2/README.md) replaces the
private, non-repeatable Pi trajectory. It copies a frozen Python repository,
gives the model two linked editing tasks in one conversation, allows file and
test tools, and runs hidden acceptance tests after each task. The coding run
used the same immutable image and policy as the source-review run, before the
permanent service switch. Its
[summary](benchmarks/runs/2026-09-29-production/summary.json) records the
fixture and runner hashes so the same workload can be repeated for a future
configuration. This is one adaptive session, not a multi-run distribution.

| Coding workload result | Measured value |
|---|---:|
| Tasks / hidden acceptance tests | **2/2 tasks, 8/8 tests passed** |
| End-to-end time | **4 min 51 s** |
| Model requests / tool calls | **14 / 18** |
| Actual input context range | **716–23,709 tokens** |
| Logical prompt / generated tokens | 155,872 / 19,507 |
| Newly computed / prefix-cached prompt tokens | 52,704 / 103,168 |
| Prefix-cache hit rate | **66.2%** |
| Weighted native prefill compute | **1,864.9 tok/s** |
| Weighted native decode | **74.5 tok/s** |
| MTP accepted / drafted tokens | **63.7%** |
| Preemptions | **0** |

The coding rates exclude tool execution; the end-to-end time includes it.
Generated tokens include reasoning. The session kept prefix caching enabled
between agent turns. Raw generated code and conversation records remain local.

## Install and serve

Use Linux with an Intel `xe` driver, Docker, access to the B70 render node,
and Python 3.10+. The frozen image build is defined in
[`docker/onednn-prefill/build-image.sh`](docker/onednn-prefill/build-image.sh);
it requires the pinned W4A8 base image and oneDNN 3.13 installation and
verifies their hashes. The production launcher will reject a missing or
incompatible image rather than switching to another engine.

```bash
git clone https://github.com/Danmoreng/intel-b70-qwen38-vllm.git
cd intel-b70-qwen38-vllm
VLLM_IMAGE=local/qwen38-b70-vllm:vllm-0.30.0-xpu-kernels-0.1.15.4 \
  ./scripts/build-image.sh
docker build --pull=false -f docker/w4a8-current/Dockerfile \
  -t local/b70-qwen38-vllm:w4a8-current-20260929 docker
cp .env.example .env
./docker/onednn-prefill/build-image.sh
./scripts/download-model.sh
./scripts/set-power-limit.py
./scripts/install-user-service.sh
```

Build the vLLM base before creating `.env`, so its production image setting
does not replace the base image tag. The oneDNN build also requires the pinned
oneDNN 3.13 installation named by `B70_ONEDNN_INSTALL` (default:
`../b70-onednn-3.13-install`) and verifies the exact W4A8 base digest.

The power script targets PCI `0000:03:00.0`; adjust
[`set-power-limit.py`](scripts/set-power-limit.py) and
[`80-b70-power.rules`](systemd/80-b70-power.rules) for another address.
The default API binding is `127.0.0.1:8081`. Check the service and API with:

```bash
systemctl --user status b70-qwen38-vllm.service
curl -fsS http://127.0.0.1:8081/health
curl -fsS http://127.0.0.1:8081/v1/models
```

Use `http://127.0.0.1:8081/v1` with model name `Qwen3.8-27B` in the agent.
The [user service template](systemd/b70-qwen38-vllm.service.in) runs the
[one-policy launcher](scripts/run-server.sh). Keep the policy, model revision
and image digest aligned when rebuilding or changing the configuration.

## Repeat the benchmarks

Run on an exclusive engine with the current image and policy:

```bash
python3 scripts/current-profile-benchmark.py --container b70-qwen38-vllm --execute
python3 scripts/run-coding-benchmark.py \
  --output-root benchmark-results/coding-agent-v2-$(date +%Y%m%d-%H%M%S)
```

The source-review runner uses the frozen
[scenario plan](benchmarks/current-profile-scenarios.json) and
[public corpus](benchmarks/meaningful-corpus.json). It stores prompts,
responses and native metrics locally. The coding runner verifies the complete
[fixture manifest](benchmarks/coding-fixture/v2/manifest.json), records the
live container identity and runs the hidden acceptance tests. Use a fresh
output directory for each coding run.

## Sources and acknowledgements

- [vLLM XPU documentation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [Intel Arc Pro B70 inference cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
- [Quantized Qwen3.8-27B model](https://huggingface.co/mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)
- [Intel Compute Runtime 26.35.39758.10](https://github.com/intel/compute-runtime/releases/tag/26.35.39758.10)
- [Intel Graphics Compiler 2.41.5](https://github.com/intel/intel-graphics-compiler/releases/tag/v2.41.5)

See [NOTICE.md](NOTICE.md) for licensing and attribution.

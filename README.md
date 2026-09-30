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
| Target linear dispatch | Runtime row selection: W4A8 from **512 total rows**, W4A16 below; preserved through compilation and graph capture |
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
`sha256:ed1ebca756abb0e0832d11cd0db026dd7e86df094c6903efe7ae8afbdc290b68`
(`local/b70-qwen38-vllm:production-onednn-v2`), pinned in
[production_image.json](config/production_image.json). The launcher checks the
exact image ID, policy and image label before serving; the image verifies its
required native libraries, GPTQ wrapper and row-dispatch helper at startup.
Both benchmarks below used this image and policy under the permanent
`b70-qwen38-vllm.service`.

This release fixes a shape-specialization regression that compiled the large-row
W4A8 branch into small-row decode graphs. An opaque runtime dispatch restores
W4A16 for those calls while separately compiling the large-row quantization.
The [decode investigation](benchmarks/experiments/onednn-prefill/decode_investigation_20260930.json)
records the controlled cycle, real-operand numerical, prefill, mixed-request and
quality checks. The model weights, native operators and profile parameters
retain their previous values.

## Source-review serving benchmark

Measured **2026-09-30** on the permanent service with the
[frozen public corpus](benchmarks/meaningful-corpus.json). The
[current summary](benchmarks/runs/2026-09-30-production/summary.json) records scenario results, fixture hashes,
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
Three additional 64K resends on a fresh worker on 2026-10-01 supply the
cold/warm 64K row, because the complete run's first 64K prefix request
reused 13,312 tokens from its 16K predecessor. Those extra prompts and
payloads match the same frozen fixtures.

### One request: context sweep

Input values are token budgets. Prefill is newly computed KV tokens per native
prefill second; decode is post-first generated tokens per native decode second.
Rates and latencies are medians across waves. MTP acceptance is weighted
across drafted tokens.

| Input / output budget | Waves | Actual input | Prefill tok/s | Decode tok/s | MTP accepted | TTFT | End to end |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 / 1,024 | 5 | 479–507 | 1,694.3 | 68.1 | 47.2% | 0.30 s | 15.32 s |
| 2,048 / 1,024 | 5 | 1,992–2,047 | 2,271.4 | 73.1 | 51.4% | 0.90 s | 14.89 s |
| 4,096 / 1,024 | 5 | 4,052–4,094 | 2,118.4 | 67.4 | 45.1% | 1.94 s | 17.11 s |
| 8,192 / 1,024 | 5 | 8,167–8,186 | 2,004.6 | 63.0 | 43.9% | 4.10 s | 20.34 s |
| 16,384 / 1,024 | 5 | 16,335–16,379 | 1,816.3 | 68.5 | 50.1% | 9.04 s | 23.97 s |
| 32,768 / 1,024 | 5 | 32,704–32,762 | 1,783.6 | 62.2 | 49.8% | 18.40 s | 34.83 s |
| 65,536 / 1,024 | 3 | 65,491–65,532 | 1,560.9 | 52.5 | 49.3% | 42.04 s | 61.45 s |
| 131,072 / 1,024 | 3 | 131,034–131,070 | 1,224.0 | 44.8 | 50.3% | 107.25 s | 130.07 s |

### One to four simultaneous requests

The rates count aggregate generated tokens in sampled intervals after all
requests emitted a first token and before any finished. C1 comes from the
context sweep; C2–C4 each include three waves with different tasks. These are
separate serving load points, not paired scaling measurements.

| Input / output per request | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| 2,048 / 1,024 | 72.5 tok/s | 117.3 tok/s | 165.5 tok/s | 203.7 tok/s |
| 4,096 / 1,024 | 65.5 tok/s | 117.4 tok/s | 149.5 tok/s | 196.6 tok/s |
| 16,384 / 1,024 | 66.5 tok/s | 101.7 tok/s | 130.6 tok/s | 162.8 tok/s |

The scheduler queued some work when capacity was tight; no request was preempted.

### Prefix reuse and maximum context

| Scenario | Measured result |
|---|---|
| 16K exact resend | 13,312 / 16,382 prompt tokens cached; TTFT **9.05 s cold → 1.92–1.95 s warm** |
| 64K exact resend | 63,232 / 65,469 prompt tokens cached; TTFT **42.29 s cold → 2.00 s warm** |
| Maximum context | **199,673 input + 1,024 output**; 976.4 prefill tok/s, 37.2 decode tok/s, 204.79 s TTFT, 232.28 s end to end |

The maximum-context row is one capacity and throughput observation. The
16K/64K resends reused the same prompt on the same worker.

## Repeatable coding-agent benchmark

The [QueueKit fixture v2](benchmarks/coding-fixture/v2/README.md) copies a frozen
Python repository and gives the model two linked editing tasks in one
conversation, with file/test tools and hidden acceptance tests after each
task. This fresh run used the same permanent-service image and policy as
the source-review run. The [summary](benchmarks/runs/2026-09-30-production/summary.json) records fixture and runner
hashes. This is one adaptive session, not a multi-run distribution.

| Coding workload result | Measured value |
|---|---:|
| Tasks / hidden acceptance tests | **2/2 tasks, 8/8 tests passed** |
| End-to-end time | **7 min 32 s** |
| Model requests / tool calls | **29 / 35** |
| Actual input context range | **716–37,496 tokens** |
| Logical prompt / generated tokens | 503,534 / 31,833 |
| Newly computed / prefix-cached prompt tokens | 104,174 / 399,360 |
| Prefix-cache hit rate | **79.3%** |
| Weighted native prefill compute | **1,728.3 tok/s** |
| Weighted native decode | **81.6 tok/s** |
| MTP accepted / drafted tokens | **68.2%** |
| Preemptions | **0** |

The coding rates exclude tool execution; end-to-end time includes it.
Generated tokens include reasoning. Prefix caching remained enabled between
agent turns. Raw generated code and conversation records remain local.

## Install and serve

Use Linux with an Intel `xe` driver, Docker, access to the B70 render node,
and Python 3.10+. The frozen image build is defined in
[`docker/onednn-prefill/build-image.sh`](docker/onednn-prefill/build-image.sh)
followed by
[`build-row-dispatch-image.sh`](docker/w4a8-current/build-row-dispatch-image.sh).
These require the pinned W4A8/oneDNN base images and oneDNN 3.13 installation
and verify their hashes. The production launcher rejects a missing or
incompatible image.

```bash
git clone https://github.com/Danmoreng/intel-b70-qwen38-vllm.git
cd intel-b70-qwen38-vllm
VLLM_IMAGE=local/qwen38-b70-vllm:vllm-0.30.0-xpu-kernels-0.1.15.4 \
  ./scripts/build-image.sh
docker build --pull=false -f docker/w4a8-current/Dockerfile \
  -t local/b70-qwen38-vllm:w4a8-current-20260929 docker
cp .env.example .env
./docker/onednn-prefill/build-image.sh
./docker/w4a8-current/build-row-dispatch-image.sh
./scripts/download-model.sh
./scripts/set-power-limit.py
./scripts/install-user-service.sh
```

Build the vLLM base before creating `.env`, so its production image setting
does not replace the base image tag. The oneDNN build also requires the pinned
oneDNN 3.13 installation named by `B70_ONEDNN_INSTALL` (default:
`../b70-onednn-3.13-install`) and verifies the exact W4A8 base digest.
The row-dispatch build uses the frozen v1 image as its base and verifies the
result against `config/production_image.json`. A different rebuilt image ID
requires review and validation before updating that release manifest.

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
export B70_FROZEN_FIXTURE_ROOT=/path/to/original/run-20260923-201101-w0.00
python3 scripts/run-readme-benchmarks.py \
  --fixture-root "$B70_FROZEN_FIXTURE_ROOT" \
  --output-root benchmark-results/readme-$(date +%Y%m%d-%H%M%S)
# Restart the idle service to obtain a cold 64K prefix, then resend it twice.
systemctl --user restart b70-qwen38-vllm.service
# Wait for /health to return HTTP 200 before running the isolated requests.
python3 scripts/current-profile-benchmark.py --container b70-qwen38-vllm \
  --only prefix-64k-cold-warm --fixture-root "$B70_FROZEN_FIXTURE_ROOT" \
  --output-root benchmark-results/prefix64-$(date +%Y%m%d-%H%M%S) --execute
```

The source-review runner uses the frozen
[scenario plan](benchmarks/current-profile-scenarios.json) and
[public corpus](benchmarks/meaningful-corpus.json). It stores prompts,
responses and native metrics locally. The coding runner verifies the complete
[fixture manifest](benchmarks/coding-fixture/v2/manifest.json), records the
live container identity and runs the hidden acceptance tests. Use a fresh
output directory for each coding run.

The combined runner acquires the benchmark lock, verifies frozen fixture
hashes, starts a fresh production worker and runs the complete source-review
and coding workloads sequentially. It leaves the production service running.
The published run replays the original full-run prompt files, including both
prefix scenarios. Raw fixtures remain local; keep their namespace, prompt
bytes and request settings fixed across configurations. The timestamp gives
each result directory a unique run ID.

## Sources and acknowledgements

- [vLLM XPU documentation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [Intel Arc Pro B70 inference cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
- [Quantized Qwen3.8-27B model](https://huggingface.co/mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)
- [Intel Compute Runtime 26.35.39758.10](https://github.com/intel/compute-runtime/releases/tag/26.35.39758.10)
- [Intel Graphics Compiler 2.41.5](https://github.com/intel/intel-graphics-compiler/releases/tag/v2.41.5)

See [NOTICE.md](NOTICE.md) for licensing and attribution.

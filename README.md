# Intel Arc Pro B70: Qwen3.8-27B with vLLM

A deployment recipe for **one 32 GB Intel Arc Pro B70 at 180 W**, with
**Q128/KV32 prefill + M04 shared-KV MTP verification**, a **200,704-token**
context, vision, tool calling and automatic prefix caching. This is the
current production configuration, verified on **2026-09-19**.

The completed coding benchmark took **42 min 47 s** and averaged **54.39
decode tok/s**, **637.49 newly computed prefill tok/s** and **93.75% prefix-cache
hits** across 133 model requests, with input context growing to 143,034 tokens.
This profile serves one active sequence and targets interactive coding latency.

## Current configuration

Q128 and M04 are used together: Q128 handles eligible prefill calls; M04
handles eligible speculative verification calls. M04's Q8 refers to a packed
query tile size. The model body uses GPTQ INT4 with FP16 activations, the
**target output head stays FP16**, and the speculative draft uses INT4 for
its head and five MTP linears.

| Setting | Value applied by this recipe |
|---|---|
| Hardware | Intel Arc Pro B70, 32 GB; one GPU / TP=1 |
| Card power cap | **180 W** |
| Model | `mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` |
| Model revision | `a47b0c6f0d756bc394c4cc629d5b0ded1acc7001` |
| Target body / activations / output head | GPTQ INT4 symmetric G128 / FP16 (W4A16) / FP16 |
| Speculation | MTP, **4** speculative tokens |
| Draft | INT4 head + five INT4 MTP linears; full **248,320**-row vocabulary |
| KV cache | **FP8** |
| Maximum context | **200,704** total input + output tokens (196 Ki tokens) |
| GPU memory fraction | **0.93** |
| Scheduler | **1** sequence; **4,096** max batched tokens |
| Prefix cache | Enabled; `--mamba-cache-mode align` |
| Attention | Q128/KV32 prefill + M04 shared-KV verification; native fallback |
| vLLM / XPU kernels | `0.29.0+xpu` / `0.1.14.1` |
| Intel userspace | Compute Runtime `26.31.39395.13`; IGC `2.40.13` |
| XPU execution | Graphs enabled; expandable allocator segments; worker `spawn` |
| Chat defaults | Medium reasoning; thinking budget capped at **8,192**; default output **16,384** |
| Parsers | Tool calls: `qwen3_xml`; reasoning: `qwen3` |
| Vision | At most 1 image / 0 videos; `max_pixels=4194304` |

The settings are implemented in [`.env.example`](.env.example),
[`run-server.sh`](scripts/run-server.sh) and the
[chat defaults middleware](runtime/request_defaults.py). The launcher explicitly
sets `B70_MTP_BF16_DRAFT=1`, `B70_DRAFT_LMHEAD_INT4=1`,
`B70_DRAFT_MTP_INT4=1`, `B70_DRAFT_VOCAB_ENABLED=0`,
`B70_GPTQ_W4A8_PREFILL=0` and `B70_XPU_SINGLE_SEED_SAMPLER=0`.
The BF16-draft flag selects the model's draft-loading path; the two INT4 flags
then convert the listed draft layers.

## Install and run

Use Linux with a working Intel `xe` driver, Docker, access to the B70 render
node, `curl`, and enough disk for the container images and approximately 20 GB
of model weights. Python 3.10+ is required for the included benchmark tools.

```bash
git clone https://github.com/Danmoreng/intel-b70-qwen38-vllm.git
cd intel-b70-qwen38-vllm
cp .env.example .env
lspci -nn | grep -Ei 'VGA|Display|Intel'
ls -l /dev/dri /dev/dri/by-path
```

Edit `.env` for your device and cache location. Defaults are
`RENDER_NODE=/dev/dri/renderD128`, `ZE_AFFINITY_MASK=0` and port `8081`.
Keep the model revision and inference settings above to reproduce this profile.

### Set the power cap

The included rule and power script target PCI **`0000:03:00.0`**. If your B70
has another address, update both [`80-b70-power.rules`](systemd/80-b70-power.rules)
and [`set-power-limit.py`](scripts/set-power-limit.py) before using them.
The account must belong to the `render` group; new group membership requires
a new login. Install the scoped permission rule once:

```bash
sudo install -m 0644 systemd/80-b70-power.rules /etc/udev/rules.d/80-b70-power.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=hwmon
./scripts/set-power-limit.py
```

The script sets and verifies the 180 W default. The systemd service also checks
it before every start and refuses to start if it cannot apply the cap.

### Build, download and serve

```bash
./scripts/build-image.sh
./scripts/download-model.sh
./scripts/set-power-limit.py
./scripts/run-server.sh
```

The default image tag is `local/b70-qwen38-vllm:q128-m04-196k-180w`.
Build, download and serving scripts read the same `.env` image setting.
The download script fetches the pinned model revision; serving uses that cache
in offline mode. Wait for `Application startup complete`, then check:

```bash
curl -fsS http://127.0.0.1:8081/health
curl -fsS http://127.0.0.1:8081/v1/models
curl -fsS http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B","messages":[{"role":"user","content":"Write a Python function that returns the square of an integer."}],"temperature":1,"top_p":0.95,"top_k":20,"max_tokens":1024}'
```

Use `http://127.0.0.1:8081/v1` and model name `Qwen3.8-27B` in your client.
The API has no built-in authentication in this recipe, so the default binding
is loopback. The operator's deployment uses a LAN binding; that transport
setting does not change the inference profile.

### Start automatically

Stop the foreground server before installing the user service:

```bash
./scripts/install-user-service.sh
journalctl --user -u b70-qwen38-vllm -f
```

The installer writes the absolute repository path into the service and starts
it. To start it at boot without an interactive login:

```bash
sudo loginctl enable-linger "$USER"
```

## Current coding benchmark

**2026-09-19, same Q128 + M04 production profile at 180 W.** Pi 0.85.1 completed
two linked tasks in a TypeScript observability dashboard: preserve configuration
when rotating login credentials, then distinguish disabled and disconnected
agent-runner states in the API and UI. The follow-up kept the same conversation
and prefix cache. Sampling was temperature 1, top-p 0.95, top-k 20, medium
reasoning, an 8,192-token thinking budget and a 16,384-token output limit.

| Session result | Measured value |
|---|---:|
| End-to-end time | **42 min 47 s**, completed normally |
| Model requests / tool calls | **133 / 131** |
| Actual input-context range | **2,177–143,034 tokens** |
| Logical prompt / generated tokens | 10,124,335 / 79,308 |
| Newly computed / prefix-cached prompt tokens | 632,879 / 9,491,456 |
| Prefix-cache hit rate | **93.75%** |
| Weighted native prefill compute | **637.49 tok/s** |
| Weighted native decode | **54.39 tok/s** |
| MTP accepted / drafted tokens | **52.20%** |
| GPU card energy / average power over session | **123.99 Wh / 173.85 W** |
| Independent task acceptance | **9/9 passed** |
| Repository tests | **570 passed**, 6 host-dependent tests skipped |

Type checking, build and lint passed. The format check reported seven existing
files, unchanged from the baseline; the combined validation command therefore
returned exit code 1. Both coding tasks completed without compaction or hitting
the 60-minute, 160-request or 100,000-generated-token limits.

### Measured rates by input context

K = 1,000 tokens; bands are lower-inclusive and upper-exclusive. Rates use
summed native phase counters across the requests in each band.

| Input context | Requests | Prefill compute (tok/s) | Decode (tok/s) | Prefix hit rate | MTP acceptance |
|---|---:|---:|---:|---:|---:|
| 0–10K | 15 | 1,427.61 | 67.87 | 34.20% | 49.57% |
| 10–20K | 6 | 1,232.84 | 55.39 | 65.90% | 38.29% |
| 20–30K | 6 | 1,066.85 | 53.14 | 78.61% | 37.35% |
| 30–40K | 5 | 953.41 | 56.65 | 86.92% | 44.73% |
| 40–50K | 8 | 822.13 | 75.06 | 90.94% | 71.24% |
| 50–60K | 3 | 766.68 | 68.28 | 87.87% | 65.33% |
| 60–70K | 20 | 697.40 | 61.74 | 93.41% | 60.35% |
| 70–80K | 14 | 642.16 | 48.84 | 93.86% | 44.44% |
| 80–90K | 5 | 606.94 | 43.55 | 92.82% | 39.43% |
| 90–100K | 7 | 561.06 | 42.89 | 94.43% | 40.28% |
| 100–110K | 5 | 516.94 | 41.56 | 94.26% | 40.84% |
| 110–120K | 6 | 485.07 | 44.64 | 95.25% | 47.11% |
| 120–130K | 11 | 460.55 | 52.34 | 96.29% | 62.26% |
| 130–140K | 18 | 434.11 | 55.15 | 96.81% | 69.82% |
| 140–150K | 4 | 423.86 | 41.44 | 96.82% | 47.71% |

The final band reaches **143,034**, not 150,000 tokens. This run does not measure
the entire configured 200,704-token context window. It is one real coding
trajectory; task content, output length and MTP acceptance vary between bands.

**Prefill** counts only newly computed prompt tokens divided by native prefill
time. **Prefix hits** count cache reuse separately. **Decode** divides generated
tokens after the first token of each request by native decode time. These
weighted engine rates exclude tool execution; the end-to-end time and GPU card
energy include it. Generated token counts include reasoning.

The measurement used an exclusive endpoint and a prefix-cache reset after
warm-up. See the [methodology](benchmarks/runs/2026-09-19-production-coding/README.md),
[profile and hashes](benchmarks/runs/2026-09-19-production-coding/profile.json),
[aggregate results](benchmarks/runs/2026-09-19-production-coding/summary.json) and
[all 133 content-free request records](benchmarks/runs/2026-09-19-production-coding/requests.json).
The private application source and task transcript are not redistributed.

## Measure your deployment

For coding measurements, use an exclusive endpoint, record native `/metrics`
counter deltas around each completed request, and use the same client settings
as above. Keep prefix caching enabled across agent turns and report cache hits
separately from newly computed prefill. The published run's
[methodology and formulas](benchmarks/runs/2026-09-19-production-coding/README.md#measurement)
explain how to aggregate by actual rendered context.

An additional synthetic cold-cache sweep is available:

```bash
./scripts/run-context-benchmark.sh benchmark-results/my-host
```

It generates exact rendered contexts at 512, 8,192, 32,768, 65,536 and 131,072
tokens, performs a full-shape warm-up and five measured requests per point,
then saves client timings and native counters. Outputs are 128 tokens for the
512-input point and 512 tokens otherwise. Allow tens of minutes on an idle
server; the script checks for competing requests and requires zero prefix hits.
This is a separate workload from the coding result above.

## Pinned build contents

[`docker/Dockerfile`](docker/Dockerfile) starts from
`vllm/vllm-openai-xpu@sha256:96db42e248d48760a4937eb3d04c4878b39d13a9814efea95d510393e097a901`
and installs:

1. SHA-256-verified Intel userspace packages at the versions above.
2. MTP and prefix-cache correctness patches from cookbook commit
   `966c593a8b375c4df5173d8d07b6be4db7835fdb`.
3. Environment-gated INT4 conversions for the draft head and five MTP linears.
4. The hash-checked Q128 library and M04 library plus the combined production
   attention adapter, documented in [`docker/m04`](docker/m04/README.md).

The prebuilt attention libraries are bound to this exact vLLM/XPU ABI. Keep
the pins to reproduce the profile; changing the base, model or kernels needs
fresh validation. The build uses the deployed attention artifacts; a rebuild's
Docker image ID can differ from the measured image ID recorded with the result.

## Sources and acknowledgements

- [vLLM XPU documentation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [Intel Arc Pro B70 inference cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
- [Quantized Qwen3.8-27B model](https://huggingface.co/mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)
- [Intel Compute Runtime 26.31.39395.13](https://github.com/intel/compute-runtime/releases/tag/26.31.39395.13)
- [Intel Graphics Compiler 2.40.13](https://github.com/intel/intel-graphics-compiler/releases/tag/v2.40.13)

See [NOTICE.md](NOTICE.md) for licensing and attribution.

# Intel Arc Pro B70: Qwen3.8-27B with vLLM

A deployment recipe for **one 32 GB Intel Arc Pro B70 at 180 W**, with
**Q128/KV32 prefill + M04 shared-KV MTP verification**, a **200,704-token**
context, vision, tool calling and automatic prefix caching. This is the
current production configuration, verified on **2026-09-22**.

With four concurrent requests, aggregate decode reaches **214.17 tok/s at
2K/512**, **215.81 tok/s at 4K/1K coding** and **196.21 tok/s at 16K/512**.

The completed coding benchmark took **41 min 38 s** and averaged **56.77
decode tok/s**, **604.00 newly computed prefill tok/s** and **94.16% prefix-cache
hits** across 119 model requests, with input context growing to 145,729 tokens.
The current scheduler permits up to four active sequences and dynamically
queues requests as KV capacity tightens.

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
| Scheduler | Up to **4** sequences; **6,656** max batched tokens; full-ISL admission; watermark **0.0** |
| Prefix cache | Enabled; `--mamba-cache-mode align` |
| Attention | Q128/KV32 prefill + M04 shared-KV verification; native fallback |
| vLLM / XPU kernels | `0.29.0+xpu` / `0.1.14.1` |
| Intel userspace | Compute Runtime `26.35.39758.10`; IGC `2.41.5` |
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

## Current phase and concurrency benchmark

The current profile was measured again on 2026-09-22 without changing or
restarting the engine. MTP4, Q128/KV32 prefill, M04 shared-KV verification,
batch 6,656, C4, FP8 KV and the 180 W cap were active throughout. The run
contained 70 measured request waves and 124 completions. Every completion was
non-empty, returned exactly the requested number of tokens and ended with
`finish_reason=length`.

The tables deliberately keep three different concepts separate:

- **prefill compute** is newly computed KV tokens divided by native vLLM
  prefill time;
- **decode** is generated tokens after the first token divided by native vLLM
  decode time; and
- **end to end** is elapsed wall time in seconds, including both phases and any
  scheduler waiting.

No end-to-end duration is presented as a model token rate. Concurrent serving
is reported as aggregate decode throughput while all requests are decoding.

### Cold-cache C1 phase sweep

Each point had one discarded full-shape warm-up. Values below are medians of
five measured requests through 32K and three at 64K/128K. Prompts were exact,
prefix-cache hits were zero and output length was forced with `ignore_eos`.

| Input / output | n | Native prefill compute | Native decode | TTFT | TPOT | End to end | MTP accepted |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 / 128 | 5 | **1,778.48 tok/s** | **84.57 tok/s** | 0.292 s | 11.82 ms | 1.79 s | 55.6% |
| 2,048 / 512 | 5 | **1,625.77 tok/s** | **84.24 tok/s** | 1.265 s | 11.87 ms | 7.33 s | 64.6% |
| 4,096 / 1,024 coding | 5 | **1,527.71 tok/s** | **80.31 tok/s** | 2.690 s | 12.45 ms | 15.40 s | 63.3% |
| 8,192 / 512 | 5 | **1,466.62 tok/s** | **80.64 tok/s** | 5.596 s | 12.40 ms | 11.94 s | 65.8% |
| 16,384 / 512 | 5 | **1,366.89 tok/s** | **77.69 tok/s** | 12.007 s | 12.86 ms | 18.58 s | 66.5% |
| 32,768 / 512 | 5 | **1,208.94 tok/s** | **66.17 tok/s** | 27.137 s | 15.10 ms | 34.85 s | 59.7% |
| 65,536 / 512 | 3 | **984.03 tok/s** | **24.65 tok/s** | 66.660 s | 40.55 ms | 87.37 s | 17.3% |
| 131,072 / 512 | 3 | **661.14 tok/s** | **56.60 tok/s** | 198.371 s | 17.63 ms | 207.37 s | 69.9% |

Decode is content-sensitive because MTP acceptance is content-sensitive. The
64K median is not a typo: one repeat decoded at 68.56 tok/s, while two decoded
at 24.51–24.65 tok/s and pulled aggregate MTP acceptance down to 17.3%.
Prefill stayed stable at 982.98–984.15 tok/s. Each 128K request reached 100%
KV use and incurred one preemption; all three still completed correctly.

### Concurrent request waves

**Aggregate decode** is measured only after every request has emitted its first
token and before any request completes. Its numerator is the native
`generation_tokens_total` counter delta sampled every 250 ms. Rates use token
and sampled-time sums across five C1 requests or three C2–C4 waves. None of the
nine C2–C4 scenarios preempted.

| Workload per request | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| 2K / 512 | 86.17 tok/s | 150.44 tok/s | 186.79 tok/s | **214.17 tok/s** |
| 4K / 1K coding | 84.17 tok/s | 130.59 tok/s | 174.79 tok/s | **215.81 tok/s** |
| 16K / 512 | 81.59 tok/s | 133.24 tok/s | 151.78 tok/s | **196.21 tok/s** |

During each selected window exactly `C` requests are decoding. The C1 phase
table above reports prefill separately; it is intentionally not mixed into this
decode-scaling table. The 250 ms counter sampling limits boundary precision,
but every C2–C4 result combines thousands of native generation tokens across
three measured waves.

### Prefix cache and maximum context

The prefix-cache probes repeated a nominally 90%-shared prompt without an
excluded warm-up, so repeat 1 is cold and repeats 2–3 are warm. The table shows
what vLLM actually reused rather than claiming the nominal share.

| Prompt / output | State | Cached / computed prompt tokens | TTFT | End to end |
|---:|---|---:|---:|---:|
| 16,384 / 512 | cold | 0 / 16,384 | 11.950 s | 19.46 s |
| 16,384 / 512 | warm median | 6,656 / 9,728 | **7.462 s** | **14.99 s** |
| 65,536 / 512 | cold | 0 / 65,536 | 66.569 s | 73.05 s |
| 65,536 / 512 | warm median | 53,248 / 12,288 | **16.441 s** | **24.30 s** |

The maximum-window probe used 200,448 input plus 256 output tokens, exactly the
configured 200,704-token limit. It completed in **432.55 s**, with **427.18 s
TTFT**, 469.45 native prefill tok/s, 47.19 native decode tok/s and 69.8% MTP
acceptance. It reached 100% KV use and incurred four preemptions. This proves
the boundary is reachable; it does not imply that the boundary is a
low-latency operating point.

The content-free [run record](benchmarks/runs/2026-09-22-current-profile/README.md)
contains method, validation, ranges and machine-readable aggregates.

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

The default image tag is `local/b70-qwen38-vllm:q128-m04-196k-180w-runtime2635`.
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

## Coding workload benchmark

**2026-09-19, Q128 + M04 at 180 W, Intel Runtime 26.35.39758.10 / IGC 2.41.5.**
This workload used single-request execution with a 4,096-token scheduler
budget. Its model, kernel and agent measurements describe the coding trajectory
independently of the concurrent synthetic waves above.
Pi 0.85.1 completed
two linked tasks in a TypeScript observability dashboard: preserve configuration
when rotating login credentials, then distinguish disabled and disconnected
agent-runner states in the API and UI. The follow-up kept the same conversation
and prefix cache. Sampling was temperature 1, top-p 0.95, top-k 20, medium
reasoning, an 8,192-token thinking budget and a 16,384-token output limit.

| Session result | Measured value |
|---|---:|
| End-to-end time | **41 min 38 s**, completed normally |
| Model requests / tool calls | **119 / 139** |
| Actual input-context range | **2,177–145,729 tokens** |
| Logical prompt / generated tokens | 9,709,218 / 80,343 |
| Newly computed / prefix-cached prompt tokens | 567,202 / 9,142,016 |
| Prefix-cache hit rate | **94.16%** |
| Weighted native prefill compute | **604.00 tok/s** |
| Weighted native decode | **56.77 tok/s** |
| MTP accepted / drafted tokens | **54.55%** |
| GPU card energy / average power over session | **119.24 Wh / 171.82 W** |
| Independent task acceptance | **9/9 passed** |
| Repository tests | **565 passed**, 6 host-dependent tests skipped |

Type checking, build and lint passed. The format check reported seven existing
files, unchanged from the baseline; the combined validation command therefore
returned exit code 1. Both coding tasks completed without compaction or hitting
the 60-minute, 160-request or 100,000-generated-token limits.

### Measured rates by input context

K = 1,000 tokens; bands are lower-inclusive and upper-exclusive. Rates use
summed native phase counters across the requests in each band.

| Input context | Requests | Prefill compute (tok/s) | Decode (tok/s) | Prefix hit rate | MTP acceptance |
|---|---:|---:|---:|---:|---:|
| 0–10K | 8 | 1,435.54 | 69.07 | 12.73% | 50.84% |
| 10–20K | 8 | 1,240.88 | 55.36 | 65.10% | 38.49% |
| 20–30K | 1 | 1,077.91 | 60.33 | 83.89% | 46.73% |
| 30–40K | 1 | 979.12 | 63.68 | 83.29% | 53.33% |
| 40–50K | 8 | 836.90 | 78.82 | 90.36% | 76.20% |
| 50–60K | 12 | 746.34 | 64.76 | 91.93% | 61.52% |
| 60–70K | 17 | 692.28 | 63.51 | 93.41% | 62.96% |
| 70–80K | 15 | 644.74 | 55.53 | 93.65% | 54.18% |
| 80–90K | 3 | 600.87 | 50.82 | 92.17% | 50.43% |
| 90–100K | 2 | 557.52 | 52.35 | 90.38% | 56.53% |
| 100–110K | 3 | 521.87 | 41.54 | 94.98% | 40.09% |
| 110–120K | 4 | 489.85 | 40.45 | 94.32% | 41.08% |
| 120–130K | 9 | 451.20 | 49.36 | 96.03% | 57.67% |
| 130–140K | 17 | 431.38 | 55.53 | 96.68% | 71.01% |
| 140–150K | 11 | 413.02 | 44.54 | 96.97% | 53.69% |

The final band reaches **145,729**, not 150,000 tokens. This run does not measure
the entire configured 200,704-token context window. A separate qualification
probe processed 200,448 input tokens and correctly returned all four control
markers; it is excluded from the coding rates. The coding benchmark is one
real trajectory; task content, output length and MTP acceptance vary between bands.

**Prefill** counts only newly computed prompt tokens divided by native prefill
time. **Prefix hits** count cache reuse separately. **Decode** divides generated
tokens after the first token of each request by native decode time. These
weighted engine rates exclude tool execution; the end-to-end time and GPU card
energy include it. Generated token counts include reasoning.

The measurement used an exclusive endpoint and a prefix-cache reset after
warm-up. See the [methodology](benchmarks/runs/2026-09-19-production-coding/README.md),
[profile and hashes](benchmarks/runs/2026-09-19-production-coding/profile.json),
[aggregate results](benchmarks/runs/2026-09-19-production-coding/summary.json) and
[all 119 content-free request records](benchmarks/runs/2026-09-19-production-coding/requests.json).
The private application source and task transcript are not redistributed.

## Measure your deployment

For a full current-profile qualification, first inspect the dry plan and then
run the phase, concurrency, prefix-cache and context suite on an exclusive
endpoint:

```bash
python3 scripts/current-profile-benchmark.py
python3 scripts/current-profile-benchmark.py --execute
```

The runner validates the active C4/full-ISL container, refuses a busy engine,
constructs exact-token prompts and records native phase counters, client
latencies, scheduler transitions, MTP acceptance, card energy and content-free
correctness evidence. Allow about one hour for the default plan.

For coding-agent measurements, use an exclusive endpoint, record native
`/metrics` counter deltas around each completed request, and use the same client
settings as above. Keep prefix caching enabled across agent turns and report
cache hits separately from newly computed prefill. The published run's
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
The [build verification](benchmarks/runs/2026-09-19-production-coding/build-verification.json)
records the successful recipe build, file comparison and production smoke checks.
Its only vLLM source difference from the measured image is an additional
startup diagnostic log statement in the measured worker.

## Sources and acknowledgements

- [vLLM XPU documentation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [Intel Arc Pro B70 inference cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
- [Quantized Qwen3.8-27B model](https://huggingface.co/mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)
- [Intel Compute Runtime 26.35.39758.10](https://github.com/intel/compute-runtime/releases/tag/26.35.39758.10)
- [Intel Graphics Compiler 2.41.5](https://github.com/intel/intel-graphics-compiler/releases/tag/v2.41.5)

See [NOTICE.md](NOTICE.md) for licensing and attribution.

# Intel Arc Pro B70: Qwen3.8-27B with vLLM

A deployment recipe for **one 32 GB Intel Arc Pro B70 at 180 W**, with
**Q128/KV32 prefill + M04 shared-KV MTP verification**, a **200,704-token**
context, vision, tool calling and automatic prefix caching. This is the
current production configuration, promoted with vLLM **0.30.0** on
**2026-09-23**. The current performance table below uses sampled reviews of
frozen source files on vLLM **0.30.0**. The older vLLM 0.29 synthetic phase
sweep is preserved in its [historical run](benchmarks/runs/2026-09-22-current-profile/README.md).

With four concurrent source-review requests, fully overlapped aggregate decode
reached **185.2 tok/s at 4K/1K** and **166.7 tok/s at 16K/1K**. At 64K/1K C1,
median native decode was **54.1 tok/s** with **48.6%** weighted MTP acceptance.

An earlier real coding-agent benchmark on vLLM 0.29 took **41 min 38 s** and
averaged **56.77 decode tok/s**, **604.00 newly computed prefill tok/s** and
**94.16% prefix-cache hits** across 119 model requests, with input context
growing to 145,729 tokens. That adaptive agent run is a separate workload.
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
| vLLM / XPU kernels | `0.30.0+xpu` / `0.1.15.4` |
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

The v0.30.0 image keeps the production Q128/M04 binaries and Intel userspace.
The previous local EAGLE/Mamba-drop patch is omitted because it suppressed all
prefix reuse on this release; v0.30 already moves the written Mamba checkpoint
to the EAGLE replay boundary. See the
[paired boundary check](benchmarks/runs/2026-09-23-prefix-boundary/README.md)
and [seeded Wikipedia MTP A/B](benchmarks/runs/2026-09-23-wikipedia-mtp/README.md).

## Current sampled source-review profile: vLLM 0.30

Measured on 2026-09-23 with the production vLLM 0.30.0 image, Q128/M04,
MTP4, FP8 KV and a verified 180 W card cap. The [frozen public corpus](benchmarks/meaningful-corpus.json)
provides complete code and documentation files with concrete review and test
tasks. Sampling used temperature 1.0, top-p 0.95, top-k 20 and pinned seeds;
thinking was disabled for this fixed-length serving screen. Every request
produced exactly 1,024 tokens with `ignore_eos=true`. The short run covered
33 waves and 51 successful requests in **28 min 05 s**. The [run report](benchmarks/runs/2026-09-23-meaningful-source/README.md)
and [machine-readable summary](benchmarks/runs/2026-09-23-meaningful-source/summary.json)
contain the exact method, actual token ranges and individual-wave ranges.

The C1 table gives medians across three source-review tasks per point, except
128K with two. Input values are token budgets; actual prompts stayed just
below them. Prefill is newly computed KV tokens per native prefill second;
decode is generated tokens after the first per native decode second. MTP
acceptance is accepted draft tokens divided by drafted tokens across the
point's requests. TTFT and end-to-end time are client wall measurements.

| Input budget / output | n | Prefill tok/s | Decode tok/s | MTP accepted | TTFT | End to end |
|---:|---:|---:|---:|---:|---:|---:|
| 512 / 1,024 | 3 | 1,725.6 | 74.7 | 49.2% | 0.30 s | 13.98 s |
| 2,048 / 1,024 | 3 | 1,600.7 | 69.4 | 47.8% | 1.28 s | 16.03 s |
| 4,096 / 1,024 | 3 | 1,501.4 | 72.5 | 53.1% | 2.74 s | 16.85 s |
| 8,192 / 1,024 | 3 | 1,416.0 | 64.0 | 44.7% | 5.79 s | 21.82 s |
| 16,384 / 1,024 | 3 | 1,312.8 | 65.6 | 49.3% | 12.49 s | 28.08 s |
| 32,768 / 1,024 | 3 | 1,158.1 | 62.6 | 48.2% | 28.30 s | 44.64 s |
| 65,536 / 1,024 | 3 | 940.6 | 54.1 | 48.6% | 69.74 s | 88.64 s |
| 131,072 / 1,024 | 2 | 681.4 | 42.9 | 47.4% | 192.48 s | 216.35 s |

The three 64K tasks had **46.8–50.0% MTP acceptance** and **52.3–54.7
native decode tok/s**. The former 17.3% at 64K came from a repeated-`x`
greedy prompt on vLLM 0.29. Prompt, sampling, output length and engine version
changed, so the two runs cannot measure a version speedup or regression.
There were no preemptions in the new sweep, including at 128K.

### Concurrent source-review requests

The table uses aggregate decode only while every request has emitted its
first token and none has finished. The generation counter was sampled every
250 ms; C2–C4 rates combine two waves each. C1 values come from the sweep
above. The tasks vary between load levels, so these are serving-load points,
not paired scaling measurements.

| Input / output per request | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| 4K / 1,024 | 73.5 tok/s | 110.5 tok/s | 146.9 tok/s | **185.2 tok/s** |
| 16K / 1,024 | 65.7 tok/s | — | — | **166.7 tok/s** |

At 4K/C4 the median batch end-to-end time was **34.75 s**; at 16K/C4 it
was **76.58 s**. The scheduler briefly had one and two waiting requests,
respectively, but neither scenario preempted.

### Prefix-cache resend

An exact 16,383-token source prompt was sent twice with the same run-specific
namespace. The first request computed all tokens; the second reused 13,312.

| State | Cached / computed prompt tokens | TTFT | End to end |
|---|---:|---:|---:|
| Cold | 0 / 16,383 | 12.54 s | 29.69 s |
| Warm | 13,312 / 3,071 | 2.53 s | 18.65 s |

The [historical vLLM 0.29 synthetic profile](benchmarks/runs/2026-09-22-current-profile/README.md)
and its [summary](benchmarks/runs/2026-09-22-current-profile/summary.json)
remain available. The current source-review run does not replace the separate
real-agent coding benchmark or the maximum-context capacity test.

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

The default image tag is `local/b70-qwen38-vllm:vllm-0.30.0-xpu-kernels-0.1.15.4`.
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
separately from the source-review serving sweep above.
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

To repeat the roughly 30-minute source-review screen shown above on an
exclusive engine:

```bash
python3 scripts/current-profile-benchmark.py \
  --scenarios benchmarks/meaningful-short-scenarios.json --execute
```

For a full current-profile qualification, first inspect the dry plan and then
run the phase, concurrency, prefix-cache and context suite on an exclusive
endpoint:

```bash
python3 scripts/current-profile-benchmark.py
python3 scripts/current-profile-benchmark.py --execute
```

The runner validates the active C4/full-ISL container, refuses a busy engine,
constructs prompts from a [frozen public source corpus](benchmarks/meaningful-benchmark.md),
and records native phase counters, client latencies, scheduler transitions,
MTP acceptance, card energy, prompt hashes and locally inspectable answers.
It uses the production coding sampler (temperature 1, top-p 0.95, top-k 20)
with thinking disabled for the short fixed-cap requests. Source files are not
padded to an exact token count: each record contains its actual prompt and
completion counts. Outputs are forced to 1,024 tokens with `ignore_eos=true`.
The full plan takes much longer than a short A/B test; use
`--only` to select a few scenarios.

For coding-agent measurements, use an exclusive endpoint, record native
`/metrics` counter deltas around each completed request, and use the agent's
documented sampling and reasoning settings. Keep prefix caching enabled across
agent turns and report cache hits separately from newly computed prefill. The published run's
[methodology and formulas](benchmarks/runs/2026-09-19-production-coding/README.md#measurement)
explain how to aggregate by actual rendered context.

An additional short cold-cache source-review sweep is available:

```bash
./scripts/run-context-benchmark.sh benchmark-results/my-host
```

It generates frozen source-review prompts under token budgets of 512, 8,192,
32,768 and 65,536, performs a full-shape warm-up and two measured requests per
point, then saves generated text, client timings and native counters. Answers
are forced to 1,024 tokens. The script checks for competing requests and
requires zero prefix hits. This is a separate workload from the coding result
above.

## Pinned build contents

[`docker/Dockerfile`](docker/Dockerfile) starts from
`vllm/vllm-openai-xpu@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d`
and installs:

1. SHA-256-verified Intel userspace packages at the versions above.
2. MTP and prefix-cache correctness patches from cookbook commit
   `966c593a8b375c4df5173d8d07b6be4db7835fdb`.
3. Environment-gated INT4 conversions for the draft head and five MTP linears.
4. The hash-checked Q128 library and M04 library plus the combined production
   attention adapter, documented in [`docker/m04`](docker/m04/README.md).
5. The SHA-256-verified published XPU kernel wheel `0.1.15.4`.

Run [`scripts/build-image.sh`](scripts/build-image.sh) to fetch and verify the
wheel before Docker builds the image. The wheel is stored only in the ignored
local build context.

The prebuilt attention libraries are bound to this exact vLLM/XPU ABI. Keep
the pins to reproduce the profile; changing the base, model or kernels needs
fresh validation. The build uses the deployed attention artifacts; a rebuild's
Docker image ID can differ from the measured image ID recorded with the result.
The [earlier build verification](benchmarks/runs/2026-09-19-production-coding/build-verification.json)
records the vLLM 0.29 recipe build and production smoke checks. The
[XPU kernel A/B and qualification](benchmarks/runs/2026-09-23-xpu-kernels-0115/README.md)
records the 0.30 wheel update; it showed no stable performance gain.

## Sources and acknowledgements

- [vLLM XPU documentation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [Intel Arc Pro B70 inference cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
- [Quantized Qwen3.8-27B model](https://huggingface.co/mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)
- [Intel Compute Runtime 26.35.39758.10](https://github.com/intel/compute-runtime/releases/tag/26.35.39758.10)
- [Intel Graphics Compiler 2.41.5](https://github.com/intel/intel-graphics-compiler/releases/tag/v2.41.5)

See [NOTICE.md](NOTICE.md) for licensing and attribution.

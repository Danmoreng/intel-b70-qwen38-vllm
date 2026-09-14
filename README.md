# Intel Arc Pro B70: fast Qwen3.8-27B serving with vLLM

This is a small, reproducible recipe for serving Qwen3.8-27B on one 32 GB
Intel Arc Pro B70. The current validated profile runs at a fixed **180 W** card
limit and combines a Q128/KV32 prefill extension with a 200,704-token context,
vision, tool calling, and automatic prefix caching.

The five-run cold-context sweep measured **77.41 decode tokens/s at 8K** and
**41.85 decode tokens/s at 128K**, with native prefill rates of 1,424.85 and
665.77 tok/s respectively. The configuration exposes 213,699 KV-cache tokens
at the configured memory fraction. It uses one request at a time; this is a
single-user latency profile, not a multi-user throughput setup.

> [!IMPORTANT]
> This project patches an exact vLLM development image. Keep the pins. Treat a
> newer base image, driver, model revision, or patch as a new experimental arm.

## Validated configuration

| Component | Value |
|---|---|
| GPU | Intel Arc Pro B70, 32 GB, one card, TP=1 |
| Model | `mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` |
| Model revision | `a47b0c6f0d756bc394c4cc629d5b0ded1acc7001` |
| Target weights | GPTQ INT4, symmetric, group size 128 |
| Target computation | W4A16 (`float16` activations); W4A8 deliberately off |
| Speculative draft | MTP4; draft LM head and five draft linears converted to INT4 |
| Draft vocabulary | Full 248,320 rows |
| KV cache | FP8 |
| Context | 200,704 total tokens (196 Ki tokens) |
| GPU memory fraction | 0.93 |
| Scheduler | one sequence, 4,096 max batched tokens |
| Prefix cache | enabled, hybrid-cache mode `align` |
| Prefill attention | Q128/KV32 for the qualified Qwen shape; native fallback otherwise |
| vLLM | `0.29.0+xpu`, XPU kernels 0.1.14.1 |
| XPU userspace | Compute Runtime 26.31.39395.13, IGC 2.40.13 |
| Card power limit | 180 W, checked before every systemd start |

Why W4A16? An end-to-end coding A/B showed that W4A8 made 8K prefill 38.1%
faster but made decode 10.3% slower and increased completed-task wall time by
15.7%. The production profile therefore optimizes the phase that dominates
long coding answers. See [the measured results](benchmarks/RESULTS.md).

## Fresh 180 W context benchmark

Five measured cold-cache requests per row, after full-shape warm-up, produced
the following client-side medians with fixed 512-token outputs:

| Input context | Prefill | Decode |
|---:|---:|---:|
| 8,192 tokens | 1,419.99 tok/s | **77.41 tok/s** |
| 32,768 tokens | 1,156.10 tok/s | **70.11 tok/s** |
| 65,536 tokens | 940.65 tok/s | **57.07 tok/s** |
| 131,072 tokens | 665.12 tok/s | **41.85 tok/s** |

All 20 long-context requests had zero prefix-cache hits and completed the
forced 512-token output. A separate 512-input/128-output point reached 100.90
decode tok/s. See
[the full methodology, ranges, native counters, and raw result records](benchmarks/RESULTS.md).

These are the qualified numbers for the current 180 W deployment, not the
highest absolute throughput ever recorded in this repository. The historical
MTP6/40K sweep was faster in most rows but changed power policy, engine/kernel
versions, MTP depth, vocabulary and scheduler budget simultaneously. The
matched 196K Q128-versus-Q256 qualification isolates the attention change:
Q128 reduced TTFT by 4.32% and improved logical prompt throughput by 4.51%.
See [the matched comparison](benchmarks/runs/2026-09-14-q128-vs-q256-196k/README.md).

## Historical real-world coding-agent benchmark

A complete local coding-agent task on the previous MTP6/40K profile
investigated and fixed an incorrect
throughput graph in a separate TypeScript application, added regression tests,
ran the full quality suite, reviewed its diff, and committed the result. The
entire run used the production endpoint and was timed end to end:

| Result | Measured value |
|---|---:|
| End-to-end wall time | **41 min 37 s** |
| Model requests / tool calls | 122 / 135 |
| Logical prompt / generated tokens | 11,456,619 / 69,149 |
| Newly computed / prefix-cached prompt tokens | 630,635 / 10,825,984 |
| Prefix-cache hit rate | 94.50% |
| Weighted native prefill compute | **658.33 tok/s** |
| Weighted native decode | **48.09 tok/s** |
| Outcome | 12 files, 378 tests passed, fix committed |

The 11.46M input count includes the full logical conversation sent again on
each agent turn. Automatic prefix caching meant only 630.6K prompt tokens
actually required new KV computation. Dividing logical tokens by prefill time
would yield 11,959.74 effective tok/s, but that is a cache-amplified application
rate—not GPU prefill compute throughput. See the
[full real-world methodology and counter deltas](benchmarks/RESULTS.md#real-world-coding-agent-run-2026-09-12).

A second, shorter coding-agent task captured native counter deltas around every
individual model request. This separates engine work from tool execution and
groups requests by the prompt context sent to the model:

| Prompt context band | Requests | Prefill compute | Decode | MTP accepted/drafted |
|---:|---:|---:|---:|---:|
| 0–10K | 6 | 1,866.19 tok/s | 103.51 tok/s | 60.4% |
| 10–20K | 2 | 1,670.52 tok/s | 66.51 tok/s | 36.4% |
| 20–30K | 23 | 1,322.39 tok/s | 63.01 tok/s | 36.8% |
| 30–40K | 5 | 1,196.93 tok/s | 65.57 tok/s | 41.0% |
| 40–50K | 6 | 1,059.11 tok/s | 54.21 tok/s | 35.6% |
| 50–60K | 7 | 880.32 tok/s | 82.77 tok/s | 66.6% |
| 60–70K | 11 | 832.44 tok/s | 50.26 tok/s | 36.6% |
| 70–80K | 4 | 767.85 tok/s | 52.10 tok/s | 40.0% |

The 13 min 2 s run completed 64 model requests and 74 tool calls, changed five
files, and passed 380 tests. Its overall weighted rates were 1,110.69 prefill
compute tok/s and 59.05 decode tok/s. Tool-call time is not part of either
rate. An offline, content-free aggregate found that the active 40K draft
vocabulary covered 98.15% of tokenized structured agent output. Responses with
more out-of-vocabulary tokens did have somewhat lower MTP acceptance, but the
relationship was weak; the reduced head can contribute, yet it does not by
itself explain the roughly 40% overall real-agent acceptance. See the
[request-level methodology and full table](benchmarks/RESULTS.md#instrumented-coding-agent-follow-up-by-context-band).

## Requirements

- Linux with a working Intel `xe` kernel driver and a visible B70 render node.
- Docker with permission to access `/dev/dri`.
- Enough disk for the base image, derived image, caches, and the approximately
  20 GB model download.
- `curl` for the smoke checks; Python 3.10+ for the benchmark utility.

Identify the B70 instead of blindly assuming the device number:

```bash
lspci -nn | grep -Ei 'VGA|Display|Intel'
ls -l /dev/dri /dev/dri/by-path 2>/dev/null
```

The default is `/dev/dri/renderD128` and Level Zero device `0`. Change both in
`.env` if your host enumerates the B70 differently.

## Quick start

Clone the repository, create the local settings file, and build the pinned
image:

```bash
git clone YOUR_GITHUB_URL/intel-b70-qwen38-vllm.git
cd intel-b70-qwen38-vllm
cp .env.example .env
./scripts/build-image.sh
```

Download the pinned model revision into the Hugging Face cache:

```bash
./scripts/download-model.sh
```

Start the server in the foreground:

```bash
./scripts/set-power-limit.py
./scripts/run-server.sh
```

Wait for `Application startup complete`, then check it from another shell:

```bash
curl -fsS http://127.0.0.1:8081/health
curl -fsS http://127.0.0.1:8081/v1/models
```

The OpenAI-compatible base URL is `http://127.0.0.1:8081/v1`; the served model
name is `Qwen3.8-27B`. No API authentication is configured. Keep the loopback
binding unless you add your own authenticated reverse proxy or otherwise trust
the network.

## Configure the 180 W power limit

The service fails closed if it cannot set and verify 180 W. Install the scoped
udev rule once so members of the `render` group can write this B70's hwmon
limit, then reload the rules:

```bash
sudo install -m 0644 systemd/80-b70-power.rules /etc/udev/rules.d/80-b70-power.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=hwmon
./scripts/set-power-limit.py
```

The checked hardware path is PCI `0000:03:00.0`. Change both the rule and
`scripts/set-power-limit.py` only after identifying a different B70 address on
your host. For a foreground launch, run the power-limit script first. The
systemd unit runs it automatically before every server start.

## Run as a systemd user service

After `.env`, the image, and the model cache are ready:

```bash
./scripts/install-user-service.sh
journalctl --user -u b70-qwen38-vllm -f
```

The installer writes a unit with the repository's absolute path, then enables
and starts it. For boot without an interactive login, an administrator can
enable lingering for the intended account:

```bash
sudo loginctl enable-linger "$USER"
```

## Re-run the context benchmark

Use an idle server. The reproducible sweep generates six unique synthetic
prompts at each exact rendered length, discards one full-shape warm-up, and
measures five cold-cache requests. It covers 8K, 32K, 64K, and 128K with
fixed 512-token outputs, plus a 512-input/128-output reference point. The
harness reads both client timings and native vLLM phase counters, asserts zero
prefix-cache hits, and stores the detailed JSON results:

```bash
./scripts/run-context-benchmark.sh
```

Pass a relative output directory to name the run explicitly:

```bash
./scripts/run-context-benchmark.sh benchmark-results/my-host
```

The exact sweep takes tens of minutes on a B70 because it processes six full
requests per context length. The script aborts if another request is active,
so do not share the endpoint with an agent or application while it runs.

These synthetic numbers measure engine throughput, not answer quality. Before
deployment, also run representative coding tests, long-context retrieval tests,
vision/tool-call checks, and an acceptance-rate comparison against the full
draft vocabulary.

## What the build changes

The Dockerfile applies:

1. pinned Intel Battlemage userspace packages;
2. pinned public MTP and prefix-cache correctness patches from the B70
   inference cookbook;
3. local, environment-gated INT4 conversion for the speculative draft LM head
   and five MTP linears; and
4. the shape-gated Q128/KV32 prefill extension in `docker/q128`.

The GPTQ target model is not modified. Draft proposals can change, but each is
verified by the target. The local draft patches fail closed for TP>1 and were
validated only for one card and one active request. The Q128 adapter checks the
complete tensor signature and falls back to native attention for unsupported
calls. Its prebuilt extension is ABI-bound to the pinned base-image digest;
source and binary hash are included next to it.

## Sources and acknowledgements

- [vLLM XPU installation documentation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [Intel Arc Pro B70 inference cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
- [Quantized Qwen3.8-27B model card](https://huggingface.co/mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)
- [Intel Compute Runtime 26.31.39395.13](https://github.com/intel/compute-runtime/releases/tag/26.31.39395.13)
- [Intel Graphics Compiler 2.40.13](https://github.com/intel/intel-graphics-compiler/releases/tag/v2.40.13)

See [NOTICE.md](NOTICE.md) for licensing and attribution details.

# Intel Arc Pro B70: fast Qwen3.8-27B serving with vLLM

This is a small, reproducible recipe for serving Qwen3.8-27B on one 32 GB
Intel Arc Pro B70. The validated profile prioritizes coding-agent decode speed
while retaining a 204,800-token context, vision, tool calling, and automatic
prefix caching.

The final measured profile reaches **103.15 decode tokens/s** on a complete
coding workload (median, two runs). A separate five-run cold-context sweep
measured **88.66 decode tokens/s at 8K** and **50.54 decode tokens/s at 128K**
with fixed 512-token outputs. The configuration exposes **215,870 KV-cache
tokens** at the configured memory fraction. It uses one request at a time;
this is a single-user latency profile, not a multi-user throughput setup.

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
| Speculative draft | MTP6; draft LM head and five draft linears converted to INT4 |
| Draft vocabulary | 40,960 workload-selected rows (optional and corpus-specific) |
| KV cache | FP8 |
| Context | 204,800 total tokens |
| GPU memory fraction | 0.93 |
| Scheduler | one sequence, 6,656 max batched tokens |
| Prefix cache | enabled, hybrid-cache mode `align` |
| vLLM | `0.27.2rc1.dev77+gac7509e2b`, XPU kernels 0.1.12.3 |
| XPU userspace | Compute Runtime 26.31.39395.13, IGC 2.40.13 |

Why W4A16? An end-to-end coding A/B showed that W4A8 made 8K prefill 38.1%
faster but made decode 10.3% slower and increased completed-task wall time by
15.7%. The production profile therefore optimizes the phase that dominates
long coding answers. See [the measured results](benchmarks/RESULTS.md).

## Fresh context benchmark

Five measured cold-cache requests per row, after full-shape warm-up, produced
the following client-side medians with fixed 512-token outputs:

| Input context | Prefill | Decode |
|---:|---:|---:|
| 8,192 tokens | 1,924.94 tok/s | **88.66 tok/s** |
| 32,768 tokens | 1,535.58 tok/s | **70.81 tok/s** |
| 65,536 tokens | 1,209.07 tok/s | **69.49 tok/s** |
| 131,072 tokens | 785.40 tok/s | **50.54 tok/s** |

All 20 requests had zero prefix-cache hits. See
[the full methodology, ranges, native counters, and raw result records](benchmarks/RESULTS.md).

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

## Optional workload-tuned 40K draft vocabulary

Reducing only the speculative draft head from 248,320 to 40,960 candidate rows
improved matched 8K decode throughput by 7.9% in the first screening. The target
head remains complete and verifies every proposed token.

The exact benchmark list is intentionally not published because it was built
from private coding-agent sessions. Generate a list from your own representative,
non-sensitive assistant outputs, source code, and documentation. Run the builder
inside the derived image so no host Python packages are required:

```bash
docker run --rm \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v "$PWD:/work" -w /work \
  --entrypoint python local/b70-qwen38-vllm:2026-09 \
  scripts/build-draft-vocab.py \
  --model mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \
  --revision a47b0c6f0d756bc394c4cc629d5b0ded1acc7001 \
  --size 40960 --output /work/draft-vocab-40960.txt \
  /work/YOUR_PUBLIC_OR_PRIVATE_CORPUS
```

Set the resulting absolute path in `.env`:

```text
DRAFT_VOCAB_FILE=/absolute/path/to/draft-vocab-40960.txt
```

If this variable is omitted, the server safely uses the full INT4 draft head.
Always benchmark your own list: a poor domain match can reduce MTP acceptance
and erase the speed benefit.

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
4. optional reduced-draft-vocabulary support.

The GPTQ target model is not modified. Draft proposals can change, but each is
verified by the target. The local draft patches fail closed for TP>1 and were
validated only for one card and one active request.

## Sources and acknowledgements

- [vLLM XPU installation documentation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [Intel Arc Pro B70 inference cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
- [Quantized Qwen3.8-27B model card](https://huggingface.co/mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)
- [Intel Compute Runtime 26.31.39395.13](https://github.com/intel/compute-runtime/releases/tag/26.31.39395.13)
- [Intel Graphics Compiler 2.40.13](https://github.com/intel/intel-graphics-compiler/releases/tag/v2.40.13)

See [NOTICE.md](NOTICE.md) for licensing and attribution details.

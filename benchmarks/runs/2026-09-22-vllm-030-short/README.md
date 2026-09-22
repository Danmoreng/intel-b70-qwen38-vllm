# Short A/B: production stack on vLLM 0.30.0

Date: 2026-09-22. Raw run ID: `v030-quick-20260922`. Wall time including
server restarts and warmups: 18 minutes 52 seconds (22:52:51–23:11:43 CEST).
Raw prompts, responses, SSE streams, server logs and telemetry remain in the
ignored local `benchmark-results/v030-quick-20260922/` directory. The tracked
[`summary.json`](summary.json) contains only aggregate metrics and hashes/counts.

## Change under test

The candidate was built with the **same** production `docker/Dockerfile`,
overriding only `BASE_IMAGE` to the published XPU vLLM 0.30.0 digest
`sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d`.
Candidate image ID: `sha256:05e0981ea0a37ed82b309dbba3157f57c9a8144aadbf2bcb8746bfed70d95016`.
Control image ID: `sha256:b675d81d4e7cc63fbcd6df395965ea16ec5c4704428c81118a1618185245dd5a`.
The Dockerfile reapplied the pinned Intel Compute Runtime 26.35, IGC 2.41.5,
cookbook fixes, local INT4 draft patches, and exact Q128/KV32 and M04 binaries
plus adapter. Image build and serving succeeded without adapting the binaries.
Both libraries actually dispatched in all four arms. vLLM changed from
`0.29.0+xpu` to `0.30.0+xpu`; XPU kernels remained `0.1.14.1`, Torch remained
`2.13.0+xpu`.

## Method

Order: candidate, control, control, candidate. Each arm used a separate
compiler cache, the same model revision and serving arguments, MTP4, FP8 KV,
max context 200704, 180 W power cap, and the same 10 prompt payloads. Two
measured repeats per scenario per arm followed one warmup. The scenarios were
coding prompts with 4096 input and 1024 output tokens, at concurrency 1 and 4,
greedy sampling (`temperature=0`). There were 16 measured waves and 40
measured requests. All responses were nonempty and exactly 1024 tokens with
`finish_reason=length`; no prefix-cache hits or preemptions occurred. The
runner and summarizer are in [`../../experiments/m12-vllm-030/`](../../experiments/m12-vllm-030/).

## Result

Means across four measured waves per version and scenario:

| Scenario | Metric | vLLM 0.29 | vLLM 0.30 | 0.30 change |
| --- | --- | ---: | ---: | ---: |
| 4K/1K, C1 | Prefill compute tok/s | 1536.9 | 1532.2 | −0.3% |
| 4K/1K, C1 | Native decode tok/s | 80.1 | 81.3 | +1.5% |
| 4K/1K, C1 | Batch wall time | 15.51 s | 15.36 s | −1.0% |
| 4K/1K, C4 | Prefill compute tok/s | 485.5 | 480.6 | −1.0% |
| 4K/1K, C4 | Fully overlapped aggregate decode tok/s | 231.7 | 206.2 | **−11.0%** |
| 4K/1K, C4 | Batch wall time | 30.53 s | 34.73 s | **+13.8%** |
| 4K/1K, C4 | MTP acceptance | 59.8% | 49.3% | −10.4 percentage points |

The C1 decode values varied substantially across repeats, so this screen does
not establish a C1 gain. The C4 slowdown occurred in every measured wave; the
lower MTP acceptance accompanies it but is not yet a proven cause. The output
hashes did not match across the two *control* arms either (0 of 10 identical
greedy outputs), so exact hash divergence between versions cannot by itself
establish a 0.30 correctness regression. Semantic correctness, long context,
vision, tools, and coding quality were outside this short screen.

**Decision:** Do not promote the vLLM 0.30 candidate. Keep the production
0.29 image. The runner restored the original production image and 180 W cap;
the service passed `/health` after restoration. A next experiment should
isolate why the C4 MTP acceptance drops before considering a longer test.

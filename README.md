# Intel Arc Pro B70: Qwen3.8-27B with vLLM

The current production profile serves **EXL3 4.00 bpw on one 32 GB Intel Arc Pro B70 at 180 W**.
It preserves the OpenAI-compatible API and uses the pinned vLLM 0.30/Torch 2.13
port, native EXL3 row dispatch, guarded exact-K oneDNN prefill and rebuilt M04
shared-KV verification. The previous GPTQ image remains an independently pinned rollback.

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

The [policy](config/production_policy.json) is `cbe1755c37b888528797ca5c0f51cf9562aa4bd8cb8ee149d0cd7bb15eb78542`.
The [release image](config/production_image.json) is `sha256:c09015ce22180fbc90ef0f5070f4a7116c8d11be785067499477accf0216f21f`
(`local/b70-qwen38-vllm:production-exl3-v1`). The final benchmark used this immutable image with
its candidate alias; promotion adds an alias and does not rebuild the payload.
The launcher verifies image/policy labels, native/M04/oneDNN/profile artifacts,
checkpoint file hashes and middleware before loading the model. Compiled caches
are isolated by policy and image. The complete EXL3 source snapshot, upstream
patch and build pins are published in [engine/exl3xpu](engine/exl3xpu/README.md).

C4 moderate-context qualification is clean. C16 permits genuine pool pressure:
all 16 independent 8K + 256-token requests completed, with four preemptions/two
affected requests and 16,000 extra submitted prefill tokens. Aligned Mamba and
speculative states share the block pool; this is not 16 simultaneous 262K contexts.
The exact 261,120 + 1,024-token boundary passes without preemption. 32 different
4.2 MP images (131,164 input tokens), video limits, long-image context,
prefix extension, scheduler-confirmed abort/recovery and independent restart
pass. See the [release report](docs/EXL3_RELEASE_REPORT.md).

## Quality and optimization evidence

The same frozen short panel retains the observed EXL3 quality advantage:
original BF16 PPL 3.60453, historical supplied GPTQ PPL 3.80193/KL 0.086369,
current EXL3 precision path PPL 3.64672/KL 0.032481. These are finite-panel
checkpoint/path measurements, not a general coding ranking or an isolated
quantization-only comparison. Matched native/optimized EXL3 has 272 bit-identical
short-panel arrays. Four engineered 32K/100K/180K/262K prefixes yield suffix
PPL 1.20858 versus BF16 1.21075, KL 0.001178 and 32/32 sampled top1 agreement.
Generated histories can differ; no bit-exact-text guarantee is claimed.
[Quality scope and raw metric receipts](benchmarks/results/exl3-migration/optimized-quality-v1/README.md).

All 409 weight tensors including 8 MTP tensors reconstruct bit-exactly. Seven
linear classes, the full target head, 18 row counts, tail poisoning, large-first
compilation and supported graphs pass unchanged numerical tolerances. Large
INT8-prefill capture is unsupported and outside the frozen decode-only profile.
The brief final kernel profile did not justify a speculative rewrite.

Mixed ABBA improves incoming 49K TTFT ~36→27 s and overlapping SSE gap p95 ~3.0→2.05 s.
Repeated 103K C1 whole-attention ABBA improves decode 43.98→49.24 tok/s cold
and 43.91→49.24 warm (~12%), with matched cache residency. Histories differ,
so this is not an isolated kernel gain. The separate M04-only proof has its own
~14% / identical 512-token scope. [Measured optimization evidence](benchmarks/results/exl3-migration/optimized-performance-v1/README.md).

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

## Coding-agent benchmarks

### Short Python fixture

The [QueueKit fixture v2](benchmarks/coding-fixture/v2/README.md) copies a frozen
Python repository and gives the model two linked editing tasks in one
conversation, with file/test tools and hidden acceptance tests after each
task. This fresh run used fresh workers with the same immutable image/profile as
the source-review run. The [summary](benchmarks/runs/2026-10-02-exl3-production/summary.json) records fixture and runner
hashes. This is one adaptive session, not a multi-run distribution.

| Coding workload result | Measured value |
|---|---:|
| Tasks / hidden acceptance tests | **1/2 tasks, 7/8 tests passed** |
| End-to-end time | **7 min 36 s** |
| Model requests / tool calls | **29 / 33** |
| Actual input context range | **716–33,967 tokens** |
| Logical prompt / generated tokens | 461,149 / 27,328 |
| Newly computed / prefix-cached prompt tokens | 97,949 / 363,200 |
| Prefix-cache hit rate | **78.8%** |
| Weighted native prefill compute | **1,883.6 tok/s** |
| Weighted native decode after first token | **67.9 tok/s** |
| MTP accepted / drafted tokens | **75.7%** |
| Preemptions | **0** |

The coding rates exclude tool execution; end-to-end time includes it.
Generated tokens include reasoning. Prefix caching remained enabled between
agent turns. Raw generated code and conversation records remain local.

**Functional task failure:** `snapshot-restore-metrics`: `test_roundtrip_and_detachment (test_task2.SnapshotRestoreMetrics.test_roundtrip_and_detachment)`.

The generated output is preserved without repairs or a replacement run. The summary records the failed test output; the [release report](docs/EXL3_RELEASE_REPORT.md) records the native-attention control and the separate release decision.

### Long WebGL2 coding task

The corrected [Flappy Bird v7 assignment](benchmarks/web-coding-fixture/v7/README.md)
uses six fixed stages, deterministic physics, procedural WebGL2 graphics,
controls, responsive UI, settings and highscores, without a level editor or
replay system. Both engines use the same task/harness, seeds, sampling,
retained reasoning, 4,096-token thinking budget and 40-minute task budget.
Unmodified final outputs are independently graded against the same 54 cases.

| Result | GPTQ production v2 | Current EXL3 v1 |
|---|---:|---:|
| Wall time / task outcome | 32min 6s; complete | 24min 12s; complete |
| Requests / maximum input context | 77 / 122,078 | 66 / 90,444 |
| Frozen functional checks | 54/54 | 54/54 |
| Native prefill / decode tok/s | 1207.9 / 58.1 | 1388.7 / 53.9 |

[All measured 10K context bands and request accounting](benchmarks/runs/2026-10-02-flappybird-v7/README.md)
preserve empty bands as unmeasured. Prefill counts new KV tokens; decode counts
post-first generated tokens including reasoning. Agent histories differ, so
overall rates and task duration do not isolate engine or quantization effects.
One seed/pair does not establish a general model-quality ranking. The older
[v6 result](benchmarks/runs/2026-10-01-flappybird/README.md) remains historical;
it is not substituted for the corrected current run.

![Rates over the growing coding context](benchmarks/runs/2026-10-02-flappybird-v7/context-rates.png)

## Comparison with the previous GPTQ profile

GPTQ numbers are the published 2026-09-30 full run; EXL3 numbers are the new
complete run using identical frozen request payloads. The GPTQ full matrix was
not repeated. Its corrected v7 coding task above is a new paired measurement.
Both profiles run at 180 W; MTP depth and quantized checkpoints differ. Remaining
decode differences are shown explicitly alongside the quality/context gain.

| Actual input tokens | GPTQ prefill | EXL3 prefill | GPTQ decode | EXL3 decode | Decode change |
|---:|---:|---:|---:|---:|---:|
| 479–507 | 1694.3 | 1700.9 | 68.1 | 58.2 | -14.6% |
| 1,992–2,047 | 2271.4 | 2422.1 | 73.1 | 59.7 | -18.3% |
| 4,052–4,094 | 2118.4 | 2235.7 | 67.4 | 55.2 | -18.2% |
| 8,167–8,186 | 2004.6 | 2300.1 | 63.0 | 54.8 | -13.1% |
| 16,335–16,379 | 1816.3 | 2166.2 | 68.5 | 56.1 | -18.0% |
| 32,704–32,762 | 1783.6 | 1974.8 | 62.2 | 56.5 | -9.2% |
| 65,491–65,532 | 1560.9 | 1668.2 | 52.5 | 52.7 | +0.3% |
| 131,034–131,070 | 1224.0 | 1261.8 | 44.8 | 39.9 | -10.9% |
| 199,673–199,673 | 976.4 | 999.0 | 37.2 | 38.5 | +3.5% |

The parallel comparison uses aggregate output only while all requests overlap.
It preserves the original task mix and payloads; generated histories and
speculative acceptance can differ between checkpoints.

| Input budget / concurrency | GPTQ aggregate decode | EXL3 aggregate decode | Change |
|---|---:|---:|---:|
| 2,048 / C2 | 117.3 | 108.5 | -7.5% |
| 2,048 / C3 | 165.5 | 152.2 | -8.1% |
| 2,048 / C4 | 203.7 | 185.6 | -8.9% |
| 4,096 / C2 | 117.4 | 110.9 | -5.6% |
| 4,096 / C3 | 149.5 | 141.4 | -5.4% |
| 4,096 / C4 | 196.6 | 181.6 | -7.7% |
| 16,384 / C2 | 101.7 | 101.5 | -0.2% |
| 16,384 / C3 | 130.6 | 137.1 | +5.0% |
| 16,384 / C4 | 162.8 | 168.2 | +3.3% |

## Install, serve and reproduce

Build instructions and immutable upstream/native/header pins are in
[engine/exl3xpu](engine/exl3xpu/README.md). This local release pins an already
qualified image; a different rebuilt image needs its own qualification and
manifest update. Model weights and large local fixtures are not redistributed.

```bash
cp .env.example .env
./scripts/download-model.sh
./scripts/set-power-limit.py
./scripts/install-user-service.sh
curl -fsS http://127.0.0.1:8081/v1/models
```

The API is `http://127.0.0.1:8081/v1`, model `Qwen3.8-27B`.
`scripts/run-server.sh` selects the frozen production profile. The independent
[GPTQ rollback launcher/config](config/releases/gptq-onednn-v2/README.md)
does not share EXL3 compiled caches. Stop the service before swapping profiles.

To repeat the full source matrix and QueueKit on the current permanent service:

```bash
python3 scripts/run-readme-benchmarks.py \
  --fixture-root /path/to/original/run-20260923-201101-w0.00 \
  --output-root benchmark-results/readme-new-run
```

The admission check reads the frozen policy (16 for this profile); the measured
matrix stays 20 scenarios / 70 waves / 124 requests at C1–C4. For an isolated
64K cold/warm resend, restart the service first and then run:

```bash
python3 scripts/current-profile-benchmark.py \
  --base http://127.0.0.1:8081 --container b70-qwen38-vllm \
  --expected-max-num-seqs 16 \
  --fixture-root /path/to/original/run-20260923-201101-w0.00 \
  --legacy-prefix-namespace --only prefix-64k-cold-warm \
  --output-root benchmark-results/prefix-64k-new-run --execute
```

The full paired release
controller and validated exports are in `scripts/run-exl3-final-readme.py`,
`summarize-readme-benchmarks.py` and `summarize-web-coding-benchmark.py`.
The large original fixtures/raw events remain local with 248 frozen file hashes.
The frozen v7 fixture README preserves its historical calibration instructions.
Its old `run-web-coding-campaign.py` entry point refuses an EXL3 production
service to prevent mislabeling it as GPTQ. Use the final release controller for
the corrected pair, or the standalone runner with the current service for a
single-engine repeat. Keep the six stages, 40-minute budget, 4K thinking budget
and fresh-worker warmup unchanged when comparing results.

On this installation, repeat the complete measured release campaign using the
retained qualification receipts and a fresh output directory:

```bash
python3 scripts/run-exl3-final-readme.py \
  --image-receipt benchmark-results/exl3-release-image-v1/image.json \
  --quality-review benchmark-results/exl3-optimized-quality-v2/quality-review.json \
  --operations-gate benchmark-results/exl3-optimized-operations-v2 \
  --performance-gate benchmark-results/exl3-optimized-performance-v1 \
  --fixture-root /path/to/original/run-20260923-201101-w0.00 \
  --out benchmark-results/exl3-repeat-new-run
systemctl --user start b70-qwen38-vllm.service
```

Run exclusively while the service is idle. The controller leaves workers off
after measuring; the last command restores the qualified current service.

## Sources and acknowledgements

- [Qwen model](https://huggingface.co/Qwen/Qwen3.8-27B)
- [EXL3 checkpoint](https://huggingface.co/turboderp/Qwen3.8-27B-exl3)
- [0xSero EXL3 XPU source](https://github.com/0xSero/exl3xpu)
- [vLLM](https://github.com/vllm-project/vllm)
- [Intel B70 cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)

See [NOTICE.md](NOTICE.md) and the preserved upstream license for attribution.

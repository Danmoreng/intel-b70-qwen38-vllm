# Q128 long-context check on production vLLM 0.30.0

Measured 2026-09-23 on the running Intel Arc Pro B70 production container
`local/qwen38-b70-vllm:vllm-0.30.0-20260923` (image SHA-256
`cd6562f03c8328fe60ca69269d0e4175a284859e56525950fbfc021be19d73f3`).
The serving run made no service or configuration change. The Q128 binary SHA-256
was `f38f23c4535407b6c3f083c5e81c89c7f704bae374571cf2e32f9da13abe0873`.

## Isolated attention operators

[`operator_sweep.py`](../../experiments/m15-q128-long-context/operator_sweep.py)
called the production Q128 binary and the native `vllm-xpu-kernels` attention
function on the same FP16 queries and randomly quantized FP8 K/V. Both received
the same causal, block-table and descale arguments. Each shape had three warmups
per arm and eight device-event timings per arm in alternating ABBA/BAAB order.
The corrected sweep uses the production interleaved K/V stride
`(3407872, 2048, 512, 1)`. The output tensors were finite, nonzero and exactly
equal at every corrected shape. Timings are medians in milliseconds. The
percentage is `(native / Q128 - 1) × 100`, so positive values favor Q128.

| New Q tokens | KV tokens | Q128 ms | Native ms | Q128 advantage | Paired wins |
|---:|---:|---:|---:|---:|---:|
| 6,656 | 16K | 60.71 | 66.21 | 9.1% | 8/8 |
| 6,656 | 64K | 288.11 | 315.43 | 9.5% | 8/8 |
| 6,656 | 128K | 599.38 | 643.80 | 7.4% | 8/8 |
| 6,656 | 192K | 918.02 | 972.96 | 6.0% | 8/8 |

Corrected raw device-event samples and tensor checks are in
[`operator-q6656-production-stride-short.json`](operator-q6656-production-stride-short.json)
and [`operator-q6656-production-stride.json`](operator-q6656-production-stride.json).
Q128's added time from 64K to 128K was 311 ms and from 128K to 192K was
319 ms: near-linear growth, without a new 128K/192K cliff. The earlier
[`q512`](operator-q512.json), [`q4096`](operator-q4096.json),
[`q6656-short`](operator-q6656-short.json) and
[`q6656`](operator-q6656.json) raw files used separate contiguous K/V allocations,
which do not match the production stride. They are retained as superseded
exploratory data and must not be used for production performance claims.

## One real 192K serving request

The existing benchmark runner sent one cold 196,608-token prompt with a 32-token
output budget, C1, MTP4 and scheduler batch 6,656. It was intentionally a
short performance probe, not a response-quality test. The request completed in
359.57 s; TTFT was 358.79 s. Native counters recorded 196,608 computed
prefill tokens, zero cached tokens, 358.59 s of prefill and 548.27 computed
tokens/s. Peak running was one, peak waiting zero, peak KV usage 95.89%, and
preemptions zero. The Q128 adapter logged 31 dispatch shapes: 29 with 6,656
queries, then 1,664 and 1,920 queries, ending at 196,608 KV tokens. See
[`summary.json`](serving/run-20260923-150609-w0.00/q128-long-context-192k-c1-r1/summary.json),
[`dispatch-summary.json`](dispatch-summary.json) and the raw serving run.

The `xpu-smi` sample during this request reported 8.8% average device-wide
memory bandwidth, but it samples the whole mixed model execution. It cannot
attribute bytes or EU activity to Q128. The installed `xpu-smi -e` returned no
EU counters; `unitrace` and VTune were absent. Thus this run does **not**
measure Q128's DRAM traffic, EU occupancy, register pressure or its internal
softmax/reduction fraction. See [`gpu-during-192k-prefill.json`](gpu-during-192k-prefill.json).

## Decision

Keep Q128 enabled. The actual production chunk size wins consistently against
native attention at 128K and 192K, and there is no abnormal long-context
scaling step. Q128 is still a worthwhile *targeted profiling* candidate: a
single 192K attention call takes about 0.92 s in isolation and the model has
16 full-attention layers. A profiler with per-kernel GPU counters and access to
the SYCL-TLA implementation is needed to determine whether softmax loads,
bandwidth, occupancy or another stage limits it. The llama.cpp softmax change
is a hypothesis, not evidence of the same bottleneck here. Do not change the
Q128 tile policy or production dispatch based on this timing sweep alone.

Reproduce the serving probe only with an idle production engine:

```bash
python3 scripts/current-profile-benchmark.py --execute \
  --scenarios benchmarks/experiments/m15-q128-long-context/serving-scenario.json \
  --container qwen38-vllm-production \
  --output-root benchmarks/runs/2026-09-23-q128-long-context/serving
```

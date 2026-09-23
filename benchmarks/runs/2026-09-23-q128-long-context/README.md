# Q128 long-context check on production vLLM 0.30.0

Measured 2026-09-23 on the running Intel Arc Pro B70 production container
`local/qwen38-b70-vllm:vllm-0.30.0-20260923` (image SHA-256
`cd6562f03c8328fe60ca69269d0e4175a284859e56525950fbfc021be19d73f3`).
No service restart or configuration change was made. The Q128 binary SHA-256
was `f38f23c4535407b6c3f083c5e81c89c7f704bae374571cf2e32f9da13abe0873`.

## Isolated attention operators

[`operator_sweep.py`](../../experiments/m15-q128-long-context/operator_sweep.py)
called the production Q128 binary and the native `vllm-xpu-kernels` attention
function on the same FP16 queries and randomly quantized FP8 K/V. Both received
the same causal, block-table and descale arguments. Each shape had three warmups
per arm and eight device-event timings per arm in alternating ABBA/BAAB order.
The output tensors were finite and exactly equal at every measured shape;
the outputs were nonzero. Timings are medians in milliseconds. The percentage
is `(native / Q128 - 1) × 100`, so positive values favor Q128.

| New Q tokens | KV tokens | Q128 ms | Native ms | Q128 advantage | Paired wins |
|---:|---:|---:|---:|---:|---:|
| 512 | 16K | 6.49 | 6.73 | 3.8% | 8/8 |
| 512 | 64K | 27.81 | 29.34 | 5.5% | 7/8 |
| 512 | 128K | 53.63 | 59.04 | 10.1% | 7/8 |
| 512 | 192K | 80.91 | 88.66 | 9.6% | 7/8 |
| 4,096 | 16K | 43.42 | 42.76 | -1.5% | 2/8 |
| 4,096 | 64K | 192.68 | 198.29 | 2.9% | 8/8 |
| 4,096 | 128K | 397.15 | 398.96 | 0.5% | 3/8 |
| 4,096 | 192K | 584.17 | 598.53 | 2.5% | 4/8 |
| 6,656 | 16K | 59.88 | 65.83 | 9.9% | 8/8 |
| 6,656 | 64K | 287.62 | 315.37 | 9.6% | 8/8 |
| 6,656 | 128K | 600.47 | 643.79 | 7.2% | 8/8 |
| 6,656 | 192K | 922.27 | 974.58 | 5.7% | 8/8 |

Raw device-event samples, tensor checks, GPU memory availability and software
versions are in [`operator-q512.json`](operator-q512.json),
[`operator-q4096.json`](operator-q4096.json),
[`operator-q6656-short.json`](operator-q6656-short.json) and
[`operator-q6656.json`](operator-q6656.json). At the actual 6,656-token
production chunk size, Q128's added time from 64K to 128K was 313 ms and from
128K to 192K was 322 ms: near-linear growth, without a new 128K/192K cliff.
The 4,096-token arm has appreciable run-to-run variation; its small differences
at 128K/192K are inconclusive.

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

# Current-profile phase and concurrency benchmark

Measured on 2026-09-22 against the already-running production engine. The
service was not restarted or reconfigured. The active profile was MTP4,
Q128/KV32 prefill, M04 shared-KV verification, FP8 KV, C4, scheduler batch
6,656, full-ISL admission, watermark 0.0, prefix caching and a 180 W card cap.
See [`profile.json`](profile.json) for the exact profile.

## Scope

The run covered:

- cold-cache C1 phase points from 512 through 131,072 input tokens;
- C2, C3 and C4 waves at 2K/512, 4K/1K coding and 16K/512 per request;
- cold and warm prefix-cache probes at 16K and 64K; and
- one 200,448-input/256-output request at the exact 200,704-token limit.

The exact plan is checked in as
[`current-profile-scenarios.json`](../../current-profile-scenarios.json). The
runner is [`current-profile-benchmark.py`](../../../scripts/current-profile-benchmark.py).
Its default invocation is a dry plan; `--execute` is required to send requests.

## Measurement

Prompts were calibrated with `/tokenize`. Each ordinary phase/concurrency point
had one discarded full-shape warm-up. Measured repetitions were five through
32K, three at 64K/128K and three per concurrent configuration. The prefix
probes intentionally had no discarded warm-up so their first request is cold.

- Prefill compute rate =
  `Δrequest_prefill_kv_computed_tokens_sum / Δrequest_prefill_time_seconds_sum`.
- Decode rate = generated tokens after the first token of each request divided
  by `Δrequest_decode_time_seconds_sum`.
- Aggregate prefill = all prompt tokens in a wave divided by time until the
  last request emits its first token.
- Fully overlapped aggregate decode = native generation-counter tokens divided
  by sampled time after every request emitted its first token and before any
  request completed.
- TPOT = client time after first generated output divided by post-first tokens.
- Batch end-to-end = simultaneous release until every request in the wave
  finished.

The native generation counter was sampled every 250 ms. Concurrent decode rates
use token and sampled-time sums across all three measured waves. Inside the
selected interval exactly C requests are in decode, so the average per active
request is aggregate decode divided by C. The summary retains the broader
serving-window rate separately, but the primary README reports only the fully
overlapped result.

`Running` is a scheduler state, not a count of decode-only sequences. A running
request may still be in chunked prefill. In the 16K/C4 waves the scheduler
progressed through 1/3, 2/2, 3/1 and 4/0 Running/Waiting states. First-token
times were staggered by the 6,656-token scheduler budget, even though all four
requests fit in KV memory and completed without preemption.

## Validation

The run recorded 70 measured waves and 124 requests: 1,964,288 logical prompt
tokens and 77,696 generated tokens. All endpoint prompt counts matched the
scenario, all responses were non-empty, every response produced its exact
requested output length, every finish reason was `length`, and all native
finished-request deltas matched concurrency. There were no request failures or
cross-request token-accounting mismatches.

Synthetic prompts requested their unique control marker and returned it.
Coding prompts intentionally requested code instead of a marker; they are
therefore validated by token, stream, finish and native-accounting checks, not
by semantic or byte-equality claims. Greedy output identity is not claimed.

## Results

The original tables are preserved in the [README at the measured repository
revision](https://github.com/Danmoreng/intel-b70-qwen38-vllm/blob/a343c40/README.md#vllm-029-phase-and-concurrency-baseline).
[`summary.json`](summary.json) contains the medians, ranges, cache deltas,
scheduler observations, MTP acceptance and full-context result in a compact,
content-free format.

Fully overlapped C4 aggregate decode is 214.17 tok/s at 2K/512, 215.81 tok/s at
4K/1K coding and 196.21 tok/s at 16K/512. The corresponding average per active
request is 53.54, 53.95 and 49.05 tok/s. Aggregate prefill is already close to
saturation at C1 and stays broadly flat rather than scaling with concurrency.

The 64K point demonstrates why repeated measurements and MTP acceptance must
be retained: prefill was stable at 982.98–984.15 tok/s, while decode ranged
from 24.51 to 68.56 tok/s as acceptance changed. The 128K requests each
preempted once; the maximum-context request preempted four times. All completed.

## Reproduction

Historical note: the commands below describe the runner at the time of this
2026-09-22 measurement. The current runner uses frozen source prompts and
sampling, so its new results are not directly comparable to these synthetic
greedy numbers. Use the repository revision from this run to reproduce the
original workload.

With the repository's default container running and no other API client active:

```bash
python3 scripts/current-profile-benchmark.py
python3 scripts/current-profile-benchmark.py --execute
```

The runner refuses execution unless the live container advertises C4 plus
explicit full-ISL admission and the engine is idle. It saves raw SSE events,
Prometheus snapshots, a 250 ms scheduler timeline, content hashes, native phase
counters, card energy and container arguments under
`benchmark-results/current-profile/`. Prompt and response text are not copied
into the aggregate record.

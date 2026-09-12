# Measured results

Measurements were collected on one Intel Arc Pro B70 32 GB with one active
request. Native vLLM prefill/decode counters were used; warmups were excluded.
Results are not cross-hardware claims and should not be read as a model-quality
benchmark.

## Fresh production context sweep (2026-09-12)

This sweep used the final W4A16, MTP6, workload-tuned 40K draft-vocabulary
profile. Each row is the median of five measured cold-cache requests following
a discarded generic warm-up and a discarded full-shape warm-up. Prompt lengths
include the rendered chat template and were checked against endpoint usage.
Every measured request produced exactly 512 tokens with `ignore_eos=true`.

| Input tokens | n | Client prefill | Client decode | Decode range | Native prefill | Native decode | MTP accepted/drafted |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 5 | 1,924.94 tok/s | **88.66 tok/s** | 76.62–95.84 | 1,932.17 tok/s | 88.63 tok/s | 45.5% |
| 32,768 | 5 | 1,535.58 tok/s | **70.81 tok/s** | 68.38–84.55 | 1,538.91 tok/s | 70.75 tok/s | 45.6% |
| 65,536 | 5 | 1,209.07 tok/s | **69.49 tok/s** | 65.15–70.05 | 1,211.08 tok/s | 69.39 tok/s | 53.1% |
| 131,072 | 5 | 785.40 tok/s | **50.54 tok/s** | 38.00–61.34 | 786.32 tok/s | 50.43 tok/s | 48.7% |

Client prefill is input tokens divided by time to first generated token. Client
decode is the remaining 511 tokens divided by time from the first generated
token to request completion. Native rates use the corresponding vLLM phase
counters. Prefix-cache hit deltas were zero for all 20 measured requests.

A report-aligned short-output point used 512 input and 128 forced output tokens:

| Input/output | n | Client prefill | Client decode | Decode range | Native prefill | Native decode | MTP accepted/drafted |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512/128 | 5 | 1,763.01 tok/s | **117.85 tok/s** | 109.74–144.64 | 1,788.27 tok/s | 117.80 tok/s | 65.6% |

The prompts span five synthetic assistant/research/RAG/tool/document families.
This exposes MTP acceptance sensitivity rather than hiding it behind one
favorable output. It also explains why these standardized synthetic decode
medians should not replace the complete coding-task result below. The exact
40K draft-vocabulary list is corpus-specific and is not published; a clone can
reproduce the method, but will only reproduce the exact profile after building
an equivalently representative local list.

The aggregate CSV and per-request JSON are in
[`runs/2026-09-12-mtp6-production`](runs/2026-09-12-mtp6-production/).

## Real-world coding-agent run (2026-09-12)

One Pi 0.85.1 coding-agent session, with medium thinking, received a real
bug-fix assignment in a separate TypeScript observability application. It had
to diagnose an implausible throughput chart end to end, correct the collector
and UI semantics, add regression tests, run the documented checks, review the
diff, write a milestone report, and commit. The endpoint was idle at the
boundary snapshots and no other model client ran during the session.

| Metric | Result |
|---|---:|
| End-to-end wall time | **2,497.248 s (41 min 37 s)** |
| Model requests | 122 |
| Agent tool calls | 135 |
| Logical prompt tokens | 11,456,619 |
| Generated tokens | 69,149 |
| Total logical tokens | 11,525,768 |
| Newly computed prompt/KV tokens | 630,635 |
| Prefix-cache hit tokens | 10,825,984 |
| Prefix-cache hit rate | **94.50%** |
| Native prefill-phase time | 957.932 s |
| Native decode-phase time | 1,435.276 s |
| Weighted prefill compute throughput | **658.33 tok/s** |
| Effective logical input / prefill time | 11,959.74 tok/s |
| Weighted decode throughput | **48.09 tok/s** |
| MTP accepted / proposed | 49,206 / 120,456 (**40.85%**) |

The rates come from vLLM counter differences captured immediately before and
after the complete agent process:

- Prefill compute = `Δrequest_prefill_kv_computed_tokens_sum ÷
  Δrequest_prefill_time_seconds_sum`.
- Decode = `(Δrequest_generation_tokens_sum − completed requests) ÷
  Δrequest_decode_time_seconds_sum`; subtracting one first token per request
  matches the post-first-token convention used by the synthetic sweep.
- Logical effective prefill = `Δrequest_prompt_tokens_sum ÷
  Δrequest_prefill_time_seconds_sum`. It is disclosed to explain the benefit
  of caching, but must not be labeled hardware prefill throughput.

The task succeeded: the agent produced a 12-file fix with 562 insertions and
55 deletions. Independent verification passed 43 contract, 282 server, and 53
web tests (378 total), plus TypeScript checks, lint, formatting, and a
production-dependency audit with zero known vulnerabilities. The fix replaced
a wall-clock delta of a batch-updated prompt counter with matched vLLM
computed-token and prefill-time histogram deltas; it did not clamp or smooth
the graph. The raw 8.3 MB agent stream and application source are not published.
A privacy-safe machine-readable record is in
[`runs/2026-09-12-real-world-coding/summary.json`](runs/2026-09-12-real-world-coding/summary.json).

## Final coding profile: MTP6 versus MTP4

Matched W4A16 runs used an 8,192-token coding prompt, up to 16,384 output
tokens, the improved workload-tuned 40K draft vocabulary, and executable tests.

| Profile | n | Median prefill | Median decode | Mean wall time | Fixed tests |
|---|---:|---:|---:|---:|---:|
| MTP4 control | 2 | 1,927.60 tok/s | 93.22 tok/s | 60.46 s | 26/26 |
| **MTP6 production** | 2 | 1,922.34 tok/s | **103.15 tok/s** | **55.61 s** | 26/26 |

MTP6 improved median decode by 10.7% and reduced mean completed-task time by
8.0%. One MTP6 response also passed all 19 self-generated tests; every output
passed the fixed suite.

## Historical throughput across context lengths

The long-context W4A16 measurements below predate the final MTP6/40K retune and
used MTP4 with the full INT4 draft head. They remain useful as a reproducible
scaling baseline; they are not mislabeled as final-MTP6 measurements.

| Input tokens | n | Prefill | Decode | Accepted/drafted |
|---:|---:|---:|---:|---:|
| 8,192 | 1 | 1,952.32 tok/s | 89.88 tok/s | 75.2% |
| 65,536 | 1 | 1,226.30 tok/s | 71.65 tok/s | 76.3% |
| 122,880 | 1 | 882.81 tok/s | 59.30 tok/s | 76.6% |

All three were complete coding requests with an output budget of 16,384. The
8K result passed 13/13 fixed and 21/21 generated tests; 65K passed 13/13 and
21/21; 122K passed 13/13 and 19/19.

## Why target W4A8 was rejected

The full coding A/B is the production decision, because prefill-only screens
hide decode regressions.

| Target path | n | Median prefill | Median decode | Mean wall time |
|---|---:|---:|---:|---:|
| **W4A16 production** | 2 | 1,935.86 tok/s | **94.99 tok/s** | **56.59 s** |
| W4A8 prefill experiment | 2 | **2,673.68 tok/s** | 85.16 tok/s | 65.47 s |

W4A8 improved prefill by 38.1%, but decode fell 10.3% and task wall time rose
15.7%. It is therefore not enabled by this repository.

The isolated cold-prefill screen explains where it may still be useful:

| Input tokens | W4A16 | W4A8 | Change |
|---:|---:|---:|---:|
| 8,192 | 1,968.19 tok/s | 2,772.53 tok/s | +40.9% |
| 65,536 | 1,226.30 tok/s | 1,497.22 tok/s | +22.1% |
| 122,880 | 882.81 tok/s | 1,014.40 tok/s | +14.9% |

At 122K, full attention accounted for 62.9% of summed GPU-kernel time, so a
faster GEMM path produces a smaller total-prefill gain as context grows.

## Draft vocabulary selection

Matched 8K/512 screens used three requests per arm:

| Draft head | Prefill | Decode | Accepted/drafted |
|---|---:|---:|---:|
| Full 248,320 rows | 1,968.19 tok/s | 104.48 tok/s | 87.7% |
| 65,536 rows | 1,967.57 tok/s | 111.04 tok/s | 88.3% |
| **40,960 rows** | 1,965.58 tok/s | **112.71 tok/s** | **89.4%** |

The selected list is corpus-dependent and is not included because the measured
version was derived from private coding sessions. The public builder lets every
user create and evaluate their own list without disclosing their corpus.

## Capacity and functional checks

- 215,870 KV-cache tokens available at `gpu-memory-utilization=0.93`.
- 204,800-token configured context passed a boundary request containing vision.
- Prefix-cache test: 17,662 prompt tokens; cold TTFT 10.915 s, warm TTFT
  2.902/2.901 s, 13,312 cached tokens reused on each warm request.
- Vision input, parsed tool call, tool-result continuation, and a real coding
  read/edit/bash smoke test passed.

Run `scripts/run-context-benchmark.sh` to generate the same privacy-safe matrix
for your exact host and final local vocabulary.

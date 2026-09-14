# Measured results

Measurements were collected on one Intel Arc Pro B70 32 GB with one active
request. Native vLLM prefill/decode counters were used; warmups were excluded.
Results are not cross-hardware claims and should not be read as a model-quality
benchmark.

## Current 180 W Q128 production sweep (2026-09-14)

The current production profile uses the Q128/KV32 prefill extension, W4A16
target computation, MTP4 with the full INT4 draft head, FP8 KV cache, a 4,096
token scheduler budget, and a 200,704-token context window. The card power
limit was verified as 180 W immediately before and after the uninterrupted
sweep. It was not continuously sampled.

Each row is the median of five measured cold-cache requests after a discarded
generic warm-up and a discarded full-shape warm-up. Prompt lengths include the
rendered chat template and were verified against endpoint usage. All 25
measured requests produced the requested fixed output with `ignore_eos=true`,
finished with `finish_reason=length`, and recorded zero prefix-cache hits.

| Input tokens | Output | n | Client prefill | Client decode | Decode range | Native prefill | Native decode | MTP accepted/drafted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 128 | 5 | 1,586.23 tok/s | **100.90 tok/s** | 91.51–107.00 | 1,676.87 tok/s | 100.92 tok/s | 77.4% |
| 8,192 | 512 | 5 | 1,419.99 tok/s | **77.41 tok/s** | 72.44–85.21 | 1,424.85 tok/s | 77.39 tok/s | 60.7% |
| 32,768 | 512 | 5 | 1,156.10 tok/s | **70.11 tok/s** | 62.32–77.76 | 1,157.94 tok/s | 70.06 tok/s | 62.5% |
| 65,536 | 512 | 5 | 940.65 tok/s | **57.07 tok/s** | 56.76–62.40 | 941.85 tok/s | 57.00 tok/s | 60.4% |
| 131,072 | 512 | 5 | 665.12 tok/s | **41.85 tok/s** | 40.35–53.15 | 665.77 tok/s | 41.77 tok/s | 55.1% |

Client prefill is input tokens divided by time to first generated token. Client
decode is the remaining output tokens divided by time after the first token.
Native rates use matching vLLM phase-counter deltas. The client and native
numbers closely agree, while the very stable native prefill rates make the
context-scaling curve especially clear.

These rows are the reproducible numbers for the new 180 W deployment, not new
all-time throughput records. Compared with the 2026-09-12 MTP6/40K sweep below,
native prefill is 6.2–26.3% lower and native decode is 1.0–17.9% lower,
depending on context. That comparison changes several variables at once:
power policy, Q128 versus native attention, vLLM/XPU-kernel versions,
MTP4/full vocabulary versus MTP6/40K, and scheduler budget. It therefore must
not be interpreted as an isolated Q128 regression or an isolated power result.
In the matched 196K Q128-versus-Q256 qualification, Q128 reduced TTFT by 4.32%
and improved logical prompt throughput by 4.51% with identical scheduler work.
The compact evidence is in
[`runs/2026-09-14-q128-vs-q256-196k`](runs/2026-09-14-q128-vs-q256-196k/).

The aggregate CSV, exact profile manifest and per-request JSON are in
[`runs/2026-09-14-q128-196k-180w`](runs/2026-09-14-q128-196k-180w/).

## Historical MTP6 production context sweep (2026-09-12)

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

## Instrumented coding-agent follow-up by context band

A second real task investigated the remaining prefix-cache query/hit rate
semantics in the same separate observability application. The agent traced the
metrics through the contract and normalization layers, removed two misleading
wall-clock-derived rates, added regression tests and documentation, ran the
quality suite, reviewed the diff, and committed the fix. It changed five files
(295 insertions, 21 deletions); independent verification passed 44 contract,
283 server, and 53 web tests (380 total), TypeScript checks, lint, formatting,
and a production-dependency audit with zero known vulnerabilities.

This run used a loopback measurement proxy that forwarded each streaming
request unchanged and retained only per-request metric deltas. It sampled the
native vLLM counters immediately before each request and after its final stream
event, before allowing the agent to begin its next request. It did not retain
prompt or response content. The engine, container, system service, and
production configuration were not changed or restarted.

The table groups requests by the complete logical prompt context of that turn.
Rates are weighted ratios of token sums to matching native phase-time sums—not
means of per-request rates:

| Prompt context band | Actual context range | n | Prefill compute | Effective logical prefill | Decode | Prefix-cache hit | MTP accepted/drafted |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–10K | 2,985–9,564 | 6 | **1,866.19 tok/s** | 2,093.16 tok/s | **103.51 tok/s** | 10.84% | 60.4% |
| 10–20K | 13,702–17,848 | 2 | **1,670.52 tok/s** | 2,648.23 tok/s | **66.51 tok/s** | 36.92% | 36.4% |
| 20–30K | 21,422–29,089 | 23 | **1,322.39 tok/s** | 7,414.74 tok/s | **63.01 tok/s** | 82.17% | 36.8% |
| 30–40K | 30,373–36,130 | 5 | **1,196.93 tok/s** | 7,028.24 tok/s | **65.57 tok/s** | 82.97% | 41.0% |
| 40–50K | 40,240–49,363 | 6 | **1,059.11 tok/s** | 6,289.12 tok/s | **54.21 tok/s** | 83.16% | 35.6% |
| 50–60K | 54,498–59,508 | 7 | **880.32 tok/s** | 11,273.68 tok/s | **82.77 tok/s** | 92.19% | 66.6% |
| 60–70K | 60,152–68,053 | 11 | **832.44 tok/s** | 11,336.00 tok/s | **50.26 tok/s** | 92.66% | 36.6% |
| 70–80K | 71,866–72,736 | 4 | **767.85 tok/s** | 11,294.01 tok/s | **52.10 tok/s** | 93.20% | 40.0% |

Overall, the run completed in **781.628 s (13 min 2 s)** with 64 model
requests and 74 tool calls. It processed 2,443,675 logical prompt tokens,
323,739 newly computed prompt tokens, 2,119,936 cached prompt tokens, and
26,489 generated tokens. The weighted results were:

- prefill compute: **1,110.69 tok/s** over 291.477 native prefill seconds;
- effective logical input: 8,383.77 tok/s (cache-amplified, not GPU compute);
- decode: **59.05 tok/s** over 447.508 native decode seconds; and
- MTP acceptance: 18,788 / 46,614 = **40.31%**.

The sum of native phase times was 738.985 s, while upstream request wall time
was 742.393 s and the complete agent wall time was 781.628 s. Tool execution
and agent orchestration therefore affect end-to-end time, but are not included
in the reported prefill or decode rates. This also resolves the ambiguity in
the longer 41-minute run: its 658.33 prefill and 48.09 decode figures are slow
native engine rates for a much longer average context, not rates diluted by
waiting for tools.

### Does the reduced 40K draft head explain the low MTP acceptance?

It may contribute, but this run does not support it as the main explanation.
Using the exact active 40,960-token ID list, an offline tokenizer-only pass over
the structured assistant output found:

| Component | Tokenized occurrences | Missing from 40K head | Coverage |
|---|---:|---:|---:|
| Thinking | 11,436 | 206 | 98.20% |
| Final text | 1,944 | 35 | 98.20% |
| Tool names and arguments | 11,164 | 214 | 98.08% |
| **Total** | **24,544** | **455** | **98.15%** |

The half of responses with fewer missing tokens had 43.44% weighted MTP
acceptance, versus 38.95% for the half with more missing tokens; the
per-response Pearson correlation between missing-token fraction and acceptance
was only −0.17. This is consistent with a modest vocabulary effect, but it is
observational and confounded by response content and length.

More importantly, the same fixed head produced 60.4% acceptance below 10K,
36–41% through most 10–50K bands, 66.6% at 50–60K, and 40.0% at 70–80K.
Coding-agent output mixes reasoning, source identifiers, paths, tool JSON, and
natural language, and this run used normal agent sampling (`temperature=1.0`,
`top_p=0.95`, `top_k=20`). The matched 8K draft-head screen used the same
sampling parameters but a fixed prompt, seed 42, short context, and 512-token
output; it reached 89.4% with the same 40K size. A decisive causal answer still
requires a matched 40K-versus-full-head A/B with identical replay prompts and
sampling; no engine profile was switched during this run.

The coverage pass deliberately excludes raw prompts/responses and protocol
wrapper tokens. Its 24,544 tokenized structured-output occurrences are
therefore slightly fewer than the 26,489 native generated-token count and
should be read as a domain-coverage diagnostic, not an exact reconstruction of
the wire token stream. The privacy-safe aggregate is in
[`runs/2026-09-12-real-world-coding-context-bands/summary.json`](runs/2026-09-12-real-world-coding-context-bands/summary.json).

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

- Current Q128/180 W profile: 213,699 KV-cache tokens available at
  `gpu-memory-utilization=0.93`.
- Current 200,704-token configured context passed an exact boundary request
  containing vision: 200,448 prompt plus 256 completion tokens, with all text,
  image and code-fix markers present.
- Current prefix-cache test: 17,664 prompt tokens; cold TTFT 13.376 s, warm
  TTFT 3.650/3.642 s, 13,312 cached tokens reused on each warm request.
- Vision input, parsed tool call and tool-result continuation passed on the
  current Q128 service.
- Historical MTP6/40K capacity was 215,870 KV-cache tokens with a 204,800-token
  boundary; its detailed functional numbers remain in Git history and the
  historical sections above.

Run `scripts/run-context-benchmark.sh` to generate the same privacy-safe matrix
for your exact host and final local vocabulary.

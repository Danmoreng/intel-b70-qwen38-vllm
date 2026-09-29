# B70 oneDNN production candidate: qualification scorecard

This scorecard records the current release gate. **The candidate is not yet
deployed as the permanent production service.** The public README will be
rewritten with current measurements only after all gates pass and deployment
is live.

## Identity

- Branch: `release/b70-onednn-v1`; integration commit `2020a8e`.
- Candidate image ID: `sha256:a42cda993bf6492acc39d23e9382e27a17efca4657bea07d80d0c28228a1623a`.
- Policy SHA-256: `4ce5f3bd77710fe08ac4b96ee761c50eb13c2d3b48ac74f20ab7fced82b4b368`.
- Model revision: `a47b0c6f0d756bc394c4cc629d5b0ded1acc7001`.
- Frozen regression fixture archive: `benchmark-results/production-release-v1/fixtures/frozen-source-prompts.tar.zst`, SHA-256 `aa45983807418efe708193814abf974ca2813c7e8293620231bdd1ce27e0b963`.
- The image entry point verified the installed oneDNN, Q128, M04 and W4A8
  binary/source hashes on both starts. The runtime AOT cache path includes the
  policy hash and image ID.

## C1 integration

- Real GPTQ dispatch test: 511 rows selected W4A16; 512 and 513 selected
  W4A8. All three outputs exactly matched their direct operators.
- Mixed numerical test: long Q256/KV16384 oneDNN plus short Q2/KV1024 M04
  matched the native batch output (`allclose` at rtol .01/atol .002,
  relative L2 `0.00013833`, maximum absolute difference `9.54e-7`). Empty
  rows, reversed order, capture exclusion and eligibility boundaries
  16383/16384/196608/196609 passed.
- Real 32K serving smoke: 32722 prompt and 32 completion tokens, 19.18 s
  client wall time; worker logs showed Q128 until KV13312 and oneDNN from
  KV19968 onward.
- Cold startup on the final image: application server began at 16:17:09 UTC
  and completed startup at 16:20:25 UTC (about 196 s); backbone
  `torch.compile` reported 94.65 s. Warm restart on the same image/cache:
  16:21:22 to 16:22:38 UTC (about 76 s); backbone compile reported 1.36 s
  and draft head compile 0.05 s. These include model loading and graph
  capture, so they are startup times, not first-request TTFT.
- A live API tool call with `add(a=2,b=3)` produced a structured
  `tool_calls` response with the correct arguments. The agent benchmark runner
  completed a two-request tool-loop smoke and recorded the live image ID.

## C2 frozen regression: complete

The `b70-release-regression-v1` autonomous unit exited successfully. Its
`final-regression/summary.json` and per-task JSONL files contain the exact
candidate image ID, prompt hashes, answers, timings and native counters.
There were zero preemptions in all 48 requests.

| Set | Passed | Sum of request wall time | Failures |
|---|---:|---:|---|
| 32K | 29/30 | 164.164 s | `context-5-structured` |
| 128K | 12/12 | 263.519 s | none |
| near-199K | 6/6 | 458.178 s | none |

The 32K failure was an arithmetic answer: for 4993 tokens at 1664 tokens per
page, the model said four pages and 1337 tokens on the last page; the correct
remainder is one token. The known 199K page-review case passed. No quality
claim is made from this regression set alone; the held-out and serving gates
below remain open.

The first coding-agent prequalification used frozen fixture v1 and made 25
model requests in 499.18 s, with 33 tool calls, 90,309 newly computed prompt
tokens, 377,728 cached prompt tokens and zero preemptions. Its recorded
acceptance scores are invalid because the runner passed absolute test paths
as unittest module names. Regrading the saved project states gives task 1
4/4 and task 2 2/4: one task 2 failure is nested payload aliasing; the other
comes from the v1 prompt omitting the JSON restore argument key. The raw run
and `coding-agent-qualification/regrade.json` remain local for audit. Fixture
v2 states `command["snapshot"]` explicitly, and the runner now uses unittest
discovery and saves each task's project state. The fresh fixture v2
prequalification finished successfully: both tasks passed 4/4 hidden
acceptance tests. It made 14 model requests and 18 tool calls in 290.832 s,
with 52,704 newly computed prompt tokens, 103,168 cached prompt tokens,
19,507 generated tokens and zero preemptions. The ignored
`coding-agent-qualification-v2/summary.json` records the exact image,
policy, fixture and runner hashes. This is a prequalification run on the
candidate service, not the final public benchmark after permanent deployment.

## Open gates

- C2: 48 new held-out tasks across at least 12 contexts, candidate/control
  pairing, new near-199K contexts and NLL diagnostics. The revised repeatable
  two-stage coding-agent fixture v2 is frozen for prequalification.
- C3: fixed 40-request integrated serving/resource trace and matched route-off
  and decode comparisons.
- C4: advertised API paths, permanent service deployment, then complete
  public source-review and repeatable coding benchmarks on the live image.
  Replace historical README measurements with those current results before
  any remote publication.

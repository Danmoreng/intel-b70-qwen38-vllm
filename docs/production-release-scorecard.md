# B70 oneDNN production candidate: qualification scorecard

The current production image is v2,
`sha256:ed1ebca756abb0e0832d11cd0db026dd7e86df094c6903efe7ae8afbdc290b68`,
pinned in `config/production_image.json` and running under the permanent
`b70-qwen38-vllm.service`. The 2026-09-30 full README run completed 20 scenarios,
70 waves and 124 requests in 53.7 minutes, without preemption or recompute.
Its fresh QueueKit v2 coding session passed 2/2 tasks and 8/8 tests in 452.03 s.
An isolated 64K cold/warm prefix run completed on 2026-10-01. Current public
numbers are in `benchmarks/runs/2026-09-30-production/summary.json`.

The row-dispatch fix and targeted T0–T3 gates are recorded in
`benchmarks/experiments/onednn-prefill/decode_investigation_20260930.json`.
Complete fixed-work cycles improved 8.03% at C1 and 9.43% at C4; frozen 32K/
128K quality added no failures. Policy, weights and native operators retain
their v1 values. T4/T5 were deferred. The user authorized production promotion
and publication of the full README benchmarks.

The sections below preserve the original v1 qualification history. On
2026-09-29 the user narrowed that closeout to the public README benchmarks.
The uncompleted C2/C3 research gates are not represented as passing.

## Historical v1 identity

- Branch: `release/b70-onednn-v1`; integration commit `2020a8e`.
- Candidate image ID: `sha256:a42cda993bf6492acc39d23e9382e27a17efca4657bea07d80d0c28228a1623a`.
- Policy SHA-256: `4ce5f3bd77710fe08ac4b96ee761c50eb13c2d3b48ac74f20ab7fced82b4b368`.
- Model revision: `a47b0c6f0d756bc394c4cc629d5b0ded1acc7001`.
- Frozen regression fixture archive: `benchmark-results/production-release-v1/fixtures/frozen-source-prompts.tar.zst`, SHA-256 `aa45983807418efe708193814abf974ca2813c7e8293620231bdd1ce27e0b963`.
- New held-out context archive: `benchmark-results/production-release-v1/fixtures/heldout-contexts-v1.tar.zst`, SHA-256 `b98440d9762a02c19709d7c234409713969ce1bf7578d39af9d7508630e9861f`; twelve contexts, including two previously unused near-199K prompt arrangements. The tracked context manifest SHA-256 is `3285e4f8edc1d8e70966e405d8e01f3b8accc31797ec2f4f37c273f27e04056c`.
- Held-out task manifest v2 SHA-256: `cac6f02aa5fd19bb5de5aae281631efbc00609904bf32d6069801ef84588a86b`; 20 code tasks (eight multi-step repository edits), twelve reviews, eight retrievals and eight required tool calls. The candidate finished 47/48; the control was stopped at 38/48 after the scope change. This is not a paired release score. The first v1 attempt stopped after 40 candidate items because its 4096-token output reservation exceeded the 200704-token context limit on the first new near-199K code task. The v1 review line and tool-marker instructions were also ambiguous, so its partial results are retained locally but are not a release score.
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
policy, fixture and runner hashes. It ran on the same immutable image and
policy as the permanent service and is reused as the current public coding
benchmark under the narrowed scope.

## Scope closeout

- C2 paired evaluation and NLL diagnostics remain incomplete because the
  control run was stopped at 38/48. The 48-item candidate result is retained
  locally and is not a passing comparison.
- C3's proposed 40-request integrated serving trace was canceled before any
  request. Its presence in the plan does not imply it was measured.
- The repeatable coding fixture v2 passed on the final image. The full public
  source-review suite passed on the same image after the permanent service
  started: 20 scenarios, 70 waves, 124 requests, zero preemptions.
- That v1 README snapshot and its machine-readable public benchmark summary
  used `benchmarks/runs/2026-09-29-production/summary.json`. The current README
  now uses the v2 measurements linked at the top of this scorecard.

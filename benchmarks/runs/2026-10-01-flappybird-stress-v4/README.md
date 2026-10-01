# Reproducible Flappy Bird coding benchmark

One fixed WebGL2 game task, one continuous conversation per engine, independent functional acceptance checks. No replay system or level editor.

| Metric | GPTQ production v2 | EXL3 4.00 bpw |
|---|---:|---:|
| Wall time (minutes) | 66.5 | 49.1 |
| Requests | 98 | 116 |
| Tool calls | 121 | 133 |
| Tool execution (minutes) | 0.1 | 0.2 |
| Card energy (Wh) | 198.7 | 146.5 |
| Maximum context | 200,509 | 141,551 |
| Native prefill (tok/s) | 910.0 | 1,118.0 |
| Native decode (tok/s) | 48.7 | 42.4 |
| Session outcome | Context window exhausted; task incomplete | All six stages attempted |
| Functional checks | 35/45 | 44/45 |
| Stages agent declared complete | 3/6 | 5/6 |
| Logical input tokens | 10,112,081 | 8,019,898 |
| Generated tokens (including reasoning) | 169,339 | 111,380 |
| Newly computed KV tokens | 444,241 | 327,098 |
| Prefix-cached tokens | 9,667,840 | 7,692,800 |
| Preemptions | 0 | 0 |
| Prefix-cache hit rate | 95.6% | 95.9% |
| MTP accepted/drafted | 51.0% | 54.4% |

## Context bands

Weighted native rates. Empty bands are unmeasured. Sparse bands and different model-generated content limit direct comparisons.

| Input context | GPTQ requests | GPTQ prefill | GPTQ decode | EXL3 requests | EXL3 prefill | EXL3 decode |
|---|---:|---:|---:|---:|---:|---:|
| 0–10K | 5 | 2,116.1 | 68.8 | 10 | 2,202.6 | 62.5 |
| 10–20K | 5 | 1,732.0 | 74.9 | 14 | 1,978.4 | 55.9 |
| 20–30K | 6 | 1,697.2 | 70.3 | 8 | 1,742.6 | 58.2 |
| 30–40K | 4 | 1,568.3 | 68.7 | 7 | 1,559.4 | 53.3 |
| 40–50K | 6 | 1,395.4 | 54.2 | 8 | 1,380.4 | 50.5 |
| 50–60K | 2 | 1,302.7 | 61.6 | 2 | 1,307.6 | 41.1 |
| 60–70K | 4 | 1,207.2 | 60.5 | 6 | 1,170.7 | 42.8 |
| 70–80K | 4 | 1,094.4 | 57.7 | 7 | 1,097.8 | 47.6 |
| 80–90K | 5 | 1,036.2 | 54.5 | 10 | 984.6 | 39.6 |
| 90–100K | 5 | 951.0 | 51.7 | 10 | 956.7 | 34.7 |
| 100–110K | 5 | 924.5 | 56.0 | 8 | 915.3 | 34.6 |
| 110–120K | 4 | 886.1 | 49.4 | 8 | 816.5 | 46.9 |
| 120–130K | 8 | 830.2 | 47.9 | 5 | 802.9 | 30.0 |
| 130–140K | 2 | 788.8 | 38.1 | 10 | 755.5 | 31.2 |
| 140–150K | 5 | 745.4 | 36.4 | 3 | 748.4 | 33.1 |
| 150–160K | 3 | 742.4 | 40.6 | 0 | — | — |
| 160–170K | 7 | 679.6 | 32.2 | 0 | — | — |
| 170–180K | 8 | 653.8 | 36.6 | 0 | — | — |
| 180–190K | 6 | 623.3 | 33.5 | 0 | — | — |
| 190–200K | 3 | 567.0 | 39.1 | 0 | — | — |
| 200–210K | 1 | 337.1 | 29.6 | 0 | — | — |

## Functional outcomes

- **GPTQ**: 35/45 functional cases passed. Failed cases: `node/app import has no DOM side effects`; `browser/WebGL2 procedural renderer and colors`; `browser/app mounts ready with accessible controls`; `browser/app buttons start pause restart`; `browser/app keyboard and canvas pointer`; `browser/app snapshot detached and invalid count`; `browser/app gameover and score/status text`; `browser/app responsive mobile canvas`; `browser/app destroy idempotent and removes input`; `browser/real entry point loads without JavaScript errors`.
  The session stopped at the shared context window: `RuntimeError: common context window exhausted: 200705`. Its wall time is time to failure, not time to finish the task. Final output was graded unmodified after stopping with the same 45 cases.
- **EXL3**: 44/45 functional cases passed. Failed cases: `browser/WebGL2 procedural renderer and colors`.

## Interpretation

The task, starter, tools, sampling and acceptance cases are frozen. The agents can choose different edits, tests and answer lengths, so their histories and MTP acceptance need not match. End-to-end time and final functional score measure the useful outcome; native rates describe the requests actually generated. This single run does not establish a general code-quality or quantization ranking.

Prefill measures newly computed tokens with the session prefix cache enabled. It does not measure cold prefill of the entire growing context. Tool execution is excluded from native phase timing and included in end-to-end time. End-to-end time also includes the CPU tokenize preflight, metric polling and fixed independent grading between stages; startup and the initial warmup are excluded. Generated reasoning remains in the conversation, consistently on both sides. Each answer has a 4,096-token thinking budget and at most 16,384 generated tokens, further bounded by a common 200,704-token input/output window.

GPTQ uses the current production-onednn-v2 profile; EXL3 uses the existing 4.00-bpw checkpoint (6-bpw output head), vLLM 0.26.1 and its existing MTP3/pruned-vocabulary recipe. GPTQ uses MTP4/full draft vocabulary and vLLM 0.30.0. Both run exclusively on the same B70 at 180 W. No claim isolates INT4 versus EXL3 from these other differences.

See comparison.json for exact image IDs, request provenance, accounting definitions, failed cases and project hashes. Generated projects, numeric request records, post-run acceptance results and compressed final model conversations are supplied alongside this report. Each saved measured request was checked against its prefix in that conversation; full repeated payloads and SSE remain local. Identical final-output grading runs after both sessions and is excluded from their elapsed time; the fixed grading between stages remains included. A context-budget failure is reported as an incomplete task and is never ranked as a faster completion.

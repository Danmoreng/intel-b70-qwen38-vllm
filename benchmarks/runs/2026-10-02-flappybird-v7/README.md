# Reproducible Flappy Bird coding benchmark

One fixed WebGL2 game task, one continuous conversation per engine, independent functional acceptance checks. No replay system or level editor.

| Metric | GPTQ production v2 | EXL3 4.00 bpw |
|---|---:|---:|
| Wall time (minutes) | 32.1 | 24.2 |
| Requests | 77 | 66 |
| Tool calls | 98 | 95 |
| Tool execution (minutes) | 0.2 | 0.2 |
| Card energy (Wh) | 95.5 | 71.9 |
| Maximum context | 122,078 | 90,444 |
| Native prefill (tok/s) | 1,207.9 | 1,388.7 |
| Native decode (tok/s) | 58.1 | 53.9 |
| Session outcome | All six stages attempted | All six stages attempted |
| Functional checks | 54/54 | 54/54 |
| Stages agent declared complete | 6/6 | 6/6 |
| Logical input tokens | 4,692,906 | 3,026,802 |
| Generated tokens (including reasoning) | 95,473 | 67,555 |
| Newly computed KV tokens | 309,930 | 242,802 |
| Prefix-cached tokens | 4,382,976 | 2,784,000 |
| Preemptions | 0 | 0 |
| Prefix-cache hit rate | 93.4% | 92.0% |
| MTP accepted/drafted | 53.2% | 59.9% |

## Context bands

Weighted native rates. Empty bands are unmeasured. Sparse bands and different model-generated content limit direct comparisons.

| Input context | GPTQ requests | GPTQ prefill | GPTQ decode | EXL3 requests | EXL3 prefill | EXL3 decode |
|---|---:|---:|---:|---:|---:|---:|
| 0–10K | 8 | 2,025.3 | 71.7 | 8 | 2,265.8 | 57.4 |
| 10–20K | 7 | 1,666.7 | 75.5 | 9 | 1,934.4 | 59.1 |
| 20–30K | 7 | 1,703.6 | 63.6 | 7 | 1,698.6 | 56.2 |
| 30–40K | 3 | 1,526.5 | 60.0 | 5 | 1,532.5 | 61.6 |
| 40–50K | 6 | 1,425.1 | 61.0 | 5 | 1,431.3 | 52.1 |
| 50–60K | 6 | 1,305.9 | 57.2 | 11 | 1,294.3 | 49.4 |
| 60–70K | 6 | 1,198.2 | 51.4 | 4 | 1,181.2 | 51.9 |
| 70–80K | 5 | 1,128.1 | 58.2 | 7 | 1,143.5 | 52.8 |
| 80–90K | 7 | 1,038.2 | 55.9 | 9 | 1,002.6 | 52.2 |
| 90–100K | 8 | 987.6 | 48.9 | 1 | 987.7 | 49.6 |
| 100–110K | 6 | 933.8 | 46.0 | 0 | — | — |
| 110–120K | 5 | 847.3 | 58.2 | 0 | — | — |
| 120–130K | 3 | 829.4 | 42.5 | 0 | — | — |

## Functional outcomes

- **GPTQ**: 54/54 functional cases passed. No failed final cases.
- **EXL3**: 54/54 functional cases passed. No failed final cases.

## Interpretation

The task, starter, tools, sampling and acceptance cases are frozen. The agents can choose different edits, tests and answer lengths, so their histories and MTP acceptance need not match. End-to-end time and final functional score measure the useful outcome; native rates describe the requests actually generated. This single run does not establish a general code-quality or quantization ranking.

Prefill measures newly computed tokens with the session prefix cache enabled. It does not measure cold prefill of the entire growing context. Tool execution is excluded from native phase timing and included in end-to-end time. End-to-end time also includes the CPU tokenize preflight, metric polling and fixed independent grading between stages; startup and the initial warmup are excluded. Generated reasoning remains in the conversation, consistently on both sides. Each answer has a 4,096-token thinking budget and at most 16,384 generated tokens, further bounded by a common 200,704-token input/output window.

GPTQ uses the current production-onednn-v2 profile; EXL3 uses the existing 4.00-bpw checkpoint (6-bpw output head), vLLM 0.26.1 and its existing MTP3/pruned-vocabulary recipe. GPTQ uses MTP4/full draft vocabulary and vLLM 0.30.0. Both run exclusively on the same B70 at 180 W. No claim isolates INT4 versus EXL3 from these other differences.

See comparison.json for exact image IDs, request provenance, accounting definitions, failed cases and project hashes. Generated projects, numeric request records, post-run acceptance results and compressed final model conversations are supplied alongside this report. Each saved measured request was checked against its prefix in that conversation; full repeated payloads and SSE remain local. Identical final-output grading is excluded from reported session elapsed time; supplemental-review-timing.json records a brief CPU-only GPTQ review that overlapped EXL3 requests 4–5; the fixed grading between stages remains included. A context-budget failure is reported as an incomplete task and is never ranked as a faster completion.

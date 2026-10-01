# Reproducible Flappy Bird coding benchmark

One fixed WebGL2 game task, one continuous conversation per engine, independent functional acceptance checks. No replay system or level editor.

| Metric | GPTQ production v2 | EXL3 4.00 bpw |
|---|---:|---:|
| Wall time (minutes) | 40.3 | 28.1 |
| Requests | 107 | 87 |
| Tool calls | 145 | 117 |
| Tool execution (minutes) | 0.1 | 0.1 |
| Card energy (Wh) | 120.3 | 83.9 |
| Maximum context | 152,762 | 106,434 |
| Native prefill (tok/s) | 1,069.5 | 1,241.6 |
| Native decode (tok/s) | 52.6 | 50.7 |
| Session outcome | Task budget exhausted; task incomplete | All six stages attempted |
| Functional checks | 51/54 | 54/54 |
| Checks after public-contract review | 52/54 | 54/54 |
| Stages agent declared complete | 4/6 | 6/6 |
| Logical input tokens | 8,381,806 | 4,674,992 |
| Generated tokens (including reasoning) | 105,148 | 74,101 |
| Newly computed KV tokens | 421,230 | 260,592 |
| Prefix-cached tokens | 7,960,576 | 4,414,400 |
| Preemptions | 0 | 0 |
| Prefix-cache hit rate | 95.0% | 94.4% |
| MTP accepted/drafted | 50.1% | 63.7% |

## Context bands

Weighted native rates. Empty bands are unmeasured. Sparse bands and different model-generated content limit direct comparisons.

| Input context | GPTQ requests | GPTQ prefill | GPTQ decode | EXL3 requests | EXL3 prefill | EXL3 decode |
|---|---:|---:|---:|---:|---:|---:|
| 0–10K | 7 | 2,034.8 | 73.8 | 8 | 2,190.3 | 62.9 |
| 10–20K | 6 | 1,690.8 | 74.1 | 11 | 2,018.1 | 59.8 |
| 20–30K | 11 | 1,701.0 | 66.0 | 6 | 1,714.3 | 54.0 |
| 30–40K | 5 | 1,548.1 | 58.6 | 6 | 1,582.2 | 54.9 |
| 40–50K | 5 | 1,381.4 | 56.5 | 13 | 1,384.2 | 47.3 |
| 50–60K | 5 | 1,294.2 | 56.7 | 2 | 1,186.5 | 56.4 |
| 60–70K | 8 | 1,207.1 | 56.2 | 6 | 1,174.8 | 46.3 |
| 70–80K | 8 | 1,111.0 | 58.9 | 13 | 1,066.6 | 41.2 |
| 80–90K | 7 | 1,029.1 | 46.9 | 8 | 1,000.5 | 43.1 |
| 90–100K | 3 | 998.5 | 53.9 | 8 | 910.8 | 49.4 |
| 100–110K | 8 | 904.1 | 43.6 | 6 | 881.8 | 44.8 |
| 110–120K | 8 | 859.2 | 41.3 | 0 | — | — |
| 120–130K | 9 | 813.3 | 40.2 | 0 | — | — |
| 130–140K | 8 | 781.0 | 40.2 | 0 | — | — |
| 140–150K | 7 | 743.0 | 51.3 | 0 | — | — |
| 150–160K | 2 | 699.4 | 38.5 | 0 | — | — |

## Functional outcomes

- **GPTQ**: 51/54 functional cases passed. Failed cases: `node/scores storage failure is reported`; `browser/settings labelled inputs and keyboard isolation`; `browser/scores save once display and survive remount`.
  The session stopped at its declared task budget: `RuntimeError: run wall-time limit`. Its wall time is time to failure, not time to finish the task. Final output was graded unmodified after stopping with the same 54 cases.
- **EXL3**: 54/54 functional cases passed. No failed final cases.

## Interpretation

The task, starter, tools, sampling and acceptance cases are frozen. The agents can choose different edits, tests and answer lengths, so their histories and MTP acceptance need not match. End-to-end time and final functional score measure the useful outcome; native rates describe the requests actually generated. This single run does not establish a general code-quality or quantization ranking.

Prefill measures newly computed tokens with the session prefix cache enabled. It does not measure cold prefill of the entire growing context. Tool execution is excluded from native phase timing and included in end-to-end time. End-to-end time also includes the CPU tokenize preflight, metric polling and fixed independent grading between stages; startup and the initial warmup are excluded. Generated reasoning remains in the conversation, consistently on both sides. Each answer has a 4,096-token thinking budget and at most 16,384 generated tokens, further bounded by a common 200,704-token input/output window.

GPTQ uses the current production-onednn-v2 profile; EXL3 uses the existing 4.00-bpw checkpoint (6-bpw output head), vLLM 0.26.1 and its existing MTP3/pruned-vocabulary recipe. GPTQ uses MTP4/full draft vocabulary and vLLM 0.30.0. Both run exclusively on the same B70 at 180 W. No claim isolates INT4 versus EXL3 from these other differences.

See comparison.json for exact image IDs, request provenance, accounting definitions, failed cases and project hashes. Generated projects, numeric request records, post-run acceptance results and compressed final model conversations are supplied alongside this report. Each saved measured request was checked against its prefix in that conversation; full repeated payloads and SSE remain local. Identical final-output grading is excluded from reported session elapsed time; supplemental-review-timing.json records a brief CPU-only GPTQ review that overlapped EXL3 requests 4–5; the fixed grading between stages remains included. A context-budget failure is reported as an incomplete task and is never ranked as a faster completion.

## Corrected contract and visual review

[QUALITY_REVIEW.md](QUALITY_REVIEW.md) explains the single storage-error harness
correction and the remaining name-input selector omission. Original grades,
post-run contract reviews, unchanged generated projects and supplemental
rendering/play evidence are supplied for both profiles. The measured fixture
was v6; corrected v7 is reserved for future repetitions.

## Identical-prompt throughput control and diagnostics

[COLD_COMPARISON.md](COLD_COMPARISON.md) supplies matched cold inputs at 103K,
139K and 188K. GPTQ has zero preemptions at all three; EXL3 has zero at 103K and
one at each longer point. The latter are explicit preempted diagnostics, not
clean zero-preemption controls. All logical/native token counts match exactly
and no input tokens are prefix-cached. Do not pool these cold observations with
the prefix-cached adaptive rates above.

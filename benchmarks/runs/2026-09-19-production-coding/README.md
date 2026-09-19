# Current production coding benchmark — 2026-09-19

One completed Pi 0.85.1 coding session on a single Arc Pro B70 at 180 W,
using **Q128 prefill + M04 verification, MTP4, GPTQ W4A16 and FP8 KV**.
The target output head is FP16; the draft head and five MTP linears are INT4.
See [the exact profile and artifact hashes](profile.json).

## Workload and outcome

The agent worked on a private TypeScript observability dashboard. Its initial
task preserved non-authentication settings and comments when rotating login
credentials, including failure/cancellation and malformed-file cases. A linked
follow-up in the same conversation distinguished disabled, disconnected and
ready agent-runner states in the API and English/German UI. The agent inspected
code, edited it, added regression tests and ran repository checks.

Both tasks completed in **2,567.412 s (42 min 47 s)** with 133 successful model
requests and 131 tool calls. There was no context compaction and no run-limit
termination (limits: 60 minutes, 160 requests, 100,000 generated tokens).
The output passed **9/9 independently held acceptance checks** and **570
repository tests**; six tests requiring unavailable host/nested-runner facilities
were skipped. Type checking, build and lint passed. The combined validation
command returned exit code 1 solely because the format check still reported
seven pre-existing files; those failures matched the frozen baseline.
These are the benchmark output's results before subsequent integration review.

The task source, prompts, generated code and tool output are private, so this
is an auditable measurement record, not a publicly replayable coding task.
The serving recipe is public; use a representative task in your own repository
for an independent coding measurement.

## Measurement

- One exclusive model endpoint, one active sequence, no competing client.
- An 8,192-input/64-output warm-up preceded a prefix-cache reset. Warm-up and
  startup are excluded from the measured session. Prefix caching then remained
  enabled throughout both tasks, including the follow-up.
- A local streaming proxy fixed temperature 1, top-p 0.95, top-k 20,
  medium reasoning, thinking budget 8,192 and maximum output 16,384 tokens.
- The benchmark engine enabled `VLLM_SERVER_DEV_MODE=1` only to expose the
  cache-reset endpoint. Normal serving does not enable this instrumentation.
- Native Prometheus counters were captured before and after every request,
  waiting for request accounting to complete. All 133 request records passed
  accounting validation. Tool execution is outside native phase timings.
- Context bands use the full rendered input prompt, with K = 1,000 tokens and
  lower-inclusive/upper-exclusive intervals. Actual prompts ranged from 2,177
  to **143,034** tokens; the 140–150K band contains no measurement above that.
- Prefill compute = sum(newly computed prompt tokens) / sum(native prefill s).
  Decode = sum(generated tokens − 1 per request) / sum(native decode s), because
  the first output token belongs to prefill. These are weighted rates, not
  averages of per-request token/s. Generated tokens include reasoning tokens.
- Prefix hit rate = sum(prefix hits) / sum(prefix queries); MTP acceptance =
  sum(accepted draft tokens) / sum(drafted tokens). Cached prompt tokens must
  not be counted as newly computed prefill work.
- Card energy integrates hwmon energy over the whole agent session, including
  tool and idle time: 446,350 J / **123.99 Wh**, averaging **173.85 W**.
  This measures the GPU card, not whole-system wall energy.

[summary.json](summary.json) contains totals and every context band.
[requests.json](requests.json) contains allowlisted numeric/timing fields and
completion statuses for all requests, without prompts, source or credentials.
Run `python3 verify.py` here to recompute the aggregate table from those records.
The main [README](../../../README.md#current-coding-benchmark) displays the results.

This is one adaptive coding trajectory with variable answer lengths and task
content. Acceptance and decode therefore vary across context bands. It is not
a cold-cache synthetic sweep or a measurement of every context up to the
configured 200,704-token limit.

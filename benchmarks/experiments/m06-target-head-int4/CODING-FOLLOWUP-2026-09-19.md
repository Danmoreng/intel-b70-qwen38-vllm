# Real-agent follow-up: investigation and proposed task

Status: the paired long run and independent checks are complete. M04 completed
both tasks; the target-head candidate exhausted the output-token budget during
the follow-up. See [the results](CODING-RESULTS-2026-09-19.md) and
[the execution harness](coding/README.md). Production was restored. No Observatory
configuration or service was changed, and no model profile has been promoted.

## Historical public results

Source: [published methodology](https://github.com/Danmoreng/intel-b70-qwen38-vllm/blob/main/benchmarks/RESULTS.md),
and the two committed summaries under `benchmarks/runs/2026-09-12-real-world-coding*`.

| Task | Wall time | Requests / tools | Native prefill | Native decode | Tests |
|---|---:|---:|---:|---:|---:|
| Correct throughput chart semantics | 41 min 37 s | 122 / 135 | 658.33 tok/s | 48.09 tok/s | 378 |
| Correct prefix query/hit rate semantics | 13 min 2 s | 64 / 74 | 1,110.69 tok/s | 59.05 tok/s | 380 |

Both used Pi 0.85.1, medium thinking, and the previous MTP6/40K profile.
Only the second public record provides per-request context bands, from
2,985 through 72,736 tokens. The first is a whole-session counter delta.
The user's recollection of a 275 W limit is consistent with the older power
policy, but neither of these public summaries records a measured power cap
or energy trace. Do not retrofit 275 W as verified per-run telemetry.

At 0–10K the second run measured 1,866.19 prefill / 103.51 decode tok/s;
at 60–70K, 832.44 / 50.26; at 70–80K, 767.85 / 52.10.
These historical numbers are workload references, not a matched INT4 comparison.

## Available instrumentation

The historical follow-up used a content-free measurement proxy that held the
request boundary until native counters were collected. A later reusable,
sandboxed real-agent implementation exists locally at:

`/home/sebastian/LocalLLM/Local-AI-B70/qwen38/optimization/dashboard-mtp-study/run-task.mjs`

It already records native counter deltas, agent events, tool time, request
usage and outcome, and launches Pi through the isolated runner. Unlike the
historical content-free proxy, this later harness also retains prompts and
responses locally; these must not be copied into public results.
It is not a drop-in M06 runner: its endpoint, runtime path, task catalogue,
limits and old profile settings need adapting before use. The 2026-09-14
`vllm029-dashboard-study` reused it for three smaller tasks; those are separate
from the two long tasks in the public README.

Read-only verification on 2026-09-19 confirmed that the running vLLM exports
the native prefill computed-token/time, generation/decode-time, request-success,
prefix-hit and accepted-speculative-token counters required for this protocol.
Model health returned HTTP 200 and the card cap was 180 W.

## A concrete new task

Use the newly observed Observatory configuration-loss bug, with a second phase
covering accurate disabled/unreachable status presentation. The task prompt is
in [CODING-TASK-CONFIG-ROTATION.md](CODING-TASK-CONFIG-ROTATION.md).
It crosses CLI configuration, regression tests, backend status and UI text,
and addresses an observed user problem. Runtime and attained context cannot
be predicted; do not pad the conversation to manufacture a 45-minute result.

Confirmed diagnosis:

- The installed Observatory service uses `dashboard/data/observatory.env`.
  That file and its running process lack `OBSERVATORY_AGENT_EXECUTION_MODE`,
  `OBSERVATORY_AGENT_RUNNER_SOCKET`, and `OBSERVATORY_AGENT_RUNNER_TOKEN`.
- The default mode is `disabled`; no RunnerAgentClient is constructed.
  The displayed safety-check message is a generic fallback, not the result
  of a failing safety probe.
- The existing isolated runner answered authenticated read-only requests:
  `/v1/snapshot` HTTP 200, `enabled=true`, `status=idle`; `/v1/projects` returned
  two projects and `/v1/tasks` returned 15 tasks. This does not prove that a
  new coding job passes every execution check.
- `configure-auth.ts` replaces the entire file when using `--force` without
  `--password-only`, dropping non-auth settings. This is a verified code path,
  not proof that this command caused the particular live incident.
- The Observatory env file was modified September 16; the runner env file
  September 13. The current service processes started September 19 at 10:37.

The minimal operational repair is to restore the three Observatory IPC settings
using `isolated`, the existing runner socket, and the existing matching control
token, then restart only Observatory and verify its authenticated agent API.
Preserve all other env entries and secret values; do not regenerate the runner
profile or reinstall services. This repair was identified, not applied.

## Proposed paired experiment

1. Freeze the same committed dashboard source and prepared dependencies for
   both arms; preserve the user's working tree. Use separate isolated copies
   with identical prompts and externally held acceptance checks. No live
   configuration, services or secrets are exposed to the coding agent.
2. A: qualified M04 image `sha256:aee9857bef1f37c8f0ee136d9f89d7166201212175a8b171d958627706cf1c0b`.
   B: repaired target-head INT4 image `sha256:790b308dfa001d3974684eb413532672ff059efc42ac277ef774a48eee43324e`.
   Both at 180 W, same model revision, MTP4/full, FP8 KV, 200,704-token limit,
   APC align, batch 4096 and one sequence. Record complete commands and IDs.
3. Pin Pi 0.85.1, medium thinking, temperature 1, top_p .95, top_k 20;
   16,384 maximum output tokens and the same 8,192 thinking budget. Align the
   client context setting with the current 200,704-token server limit.
   Record compaction separately, including its model requests.
4. Perform equivalent excluded warm-ups and start each arm with an empty
   prefix cache. Preserve natural caching throughout each coding session.
   Allow no concurrent inference client. A run ends when the task is complete;
   use a disclosed ceiling, for example 60 minutes, 160 requests and 100K
   output tokens. Record failures and limits reached instead of discarding them.
5. Capture native metrics before/after EVERY request, after final engine
   accounting is visible and before allowing the next request. Validate exactly
   one completion, no counter reset and usage/counter agreement. Flag missing
   or contaminated measurements; never replace missing counters with zero.
6. Compute weighted context bands of 10,000 logical prompt tokens:
   `sum(computed prompt tokens) / sum(native prefill seconds)` and
   `sum(generated tokens - 1 per completed request) / sum(native decode seconds)`.
   Report band counts, actual context ranges, newly computed and cached tokens,
   prefix hit rate, MTP acceptance, TTFT and phase times. Logical input divided
   by prefill time is a separate cache-amplified metric, not GPU compute speed.
7. Record card power cap, temperature and energy throughout; report actual
   joules and tokens/J where available, plus tool time and total task time.
   Independently run unchanged acceptance checks and review the produced diff.
8. Restore and verify production on every exit path. Publish only sanitized
   numerical summaries. Show unvisited context bands as unmeasured.

A first A/B pair is a practical screen. Adaptive agent trajectories can differ
despite identical initial inputs. Repeat in reversed order for stronger evidence;
use identical saved-request replay separately if isolating the head's inference
effect is required. Do not infer a pure INT4 speedup from old-versus-new tasks.

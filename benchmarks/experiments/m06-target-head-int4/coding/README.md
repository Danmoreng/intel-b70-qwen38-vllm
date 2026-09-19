# M06 real-agent coding comparison

Host-specific harness prepared and started on 2026-09-19. The run directory is
recorded in `LATEST`; the service is `b70-m06-coding-longrun.service`.
This harness does not enable or change the live Observatory Tasks feature.

The two arms use the qualified M04 control and the repaired INT4 target head,
both at 180 W. Each receives a fresh copy of dashboard revision
`3ea7cc6cd0ff9422d3bffe54a94aa3cc14506d5c`, frozen dependencies, the same initial
task and follow-up, and Pi 0.85.1 with medium thinking. Limits per arm are
60 minutes, 160 requests and 100,000 generated tokens. Completion can occur
earlier. Compaction is enabled and its events are recorded. Context is allowed
to grow naturally; absent context bands are not estimated or filled artificially.

The initial task addresses preservation of configuration during credential
rotation; the follow-up distinguishes configured-disabled and unreachable
runner states. Both execute inside the existing bubblewrap/systemd isolation.
Only a narrowly scoped model relay is exposed; live configuration and services
are outside the coding workspace. The external evaluator is mounted only for
independent checks after the agent finishes.

Before starting inference:

- The actual isolated Pi/relay protocol passed three mock model requests,
  including a real read tool and the follow-up. The mock deliberately delays
  counters by 200 ms; the harness waits for accounting before releasing DONE.
- The external acceptance suite reproduced five failures and four passes on
  unchanged source. It executes the real compiled CLI with dummy terminal
  input and fake credentials, and exercises the backend status API.
- Baseline checks passed 535 tests, builds, type checks and lint. Six host-only
  tests are skipped through the repository's existing `B70_RELEASE_BUILD=1`
  switch because nested host sandbox execution is unavailable inside a job.
  Tests use two workers to fit the unchanged sandbox process limit.
- Seven existing formatting warnings are frozen in the manifest. The evaluator
  reports them separately and does not treat a baseline-only format failure as
  a newly introduced regression. Earlier preflight failures are retained.

`run-study.py` owns the exclusive GPU lock, checks pinned image IDs and power,
warms each arm equally, resets the prefix cache, and launches `run-task.mjs`.
The relay checks native completion counters and token usage around each request,
retains raw metrics, and rejects missing/reset/contaminated accounting.
`summary.json` uses weighted token sums divided by native phase-time sums,
grouped in 10,000-token logical context bands. Task/tool time, MTP acceptance,
cache hits, compaction and card energy are reported separately.

Raw prompts, reasoning, tool output and generated source stay in the local run
directory. Public results should include sanitized numeric summaries only.
The harness, prompts and evaluator are copied and hashed before launch; both
arms execute that snapshot. Source fixes remain isolated, with no promotion,
live configuration change, deployment or push.

Production restoration runs in `finally` and in systemd `ExecStopPost`, guarded
by an ownership marker. Independent code checks run after the model experiments
release the GPU and restore production. A task failure remains a result; it does
not suppress the second arm. `complete-not-promoted` describes completion of the
experiment, not automatic acceptance of either generated fix.

Inspect progress:

```bash
systemctl --user status b70-m06-coding-longrun.service
journalctl --user -u b70-m06-coding-longrun.service
```

The run contains `state.json`, per-arm `task-result.json`, local code workspaces,
request metrics, validation logs, `summary.json`, `RESULTS.md`, and
`production-restored.json`. No whole-system wall-socket energy claim is made.
An adaptive A/B pair is a practical screen; it cannot isolate pure kernel speed
from differences in the models' coding trajectories.

The first infrastructure attempt stopped before any coding request because
vLLM 0.29 exposes cache reset only in server development mode. Both experimental
arms now set VLLM_SERVER_DEV_MODE=1 on the loopback-only engine and verify the
reset success response. The sandbox relay exposes only chat completions, so
these administrative endpoints are not available to the coding agent. Production
retains its original settings. The failed attempt and restoration proof remain
in the preceding run directory.

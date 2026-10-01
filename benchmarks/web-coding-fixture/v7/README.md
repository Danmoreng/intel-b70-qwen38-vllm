# Flappy Bird WebGL2 coding benchmark v7

A small offline browser game, not a dashboard modification. There is no level
editor, replay system, campaign, framework, third-party runtime library or
backend. The complete starter, task sequence and test harness are public and
hash-pinned by [manifest.json](manifest.json).

The agent implements six stages in one conversation:

1. Seeded random numbers and validated configuration.
2. Deterministic fixed-step flight, pipes, collision and scoring.
3. Keyboard/touch controls and fixed-step scheduling.
4. Procedural WebGL2 drawing and renderer lifecycle.
5. Playable responsive app with start, pause, restart and accessible status.
6. Validated settings, persisted local highscores and a short user guide.

[tasks.json](tasks.json) is the exact stage sequence. The public API contracts
are in [project/specs](project/specs); the agent receives each stage separately
and can inspect the whole starter. No compaction, minimum elapsed time or
artificial context-padding requirement is used. The initial game seed is
12345. The model sampling seed schedule is `73000 + request_index - 1`, reset
for each engine run.

Version 5 completed correctly at only 83K input tokens. Version 6 therefore
adds a meaningful final-stage settings and local highscore task, with dedicated
acceptance cases. It retains the small game, geometry reference, first five
stage instructions, 4K thinking budget and all previous behavior. There is no
editor or replay. Both engines start fresh; earlier results remain separate.

## Common request settings

- Temperature 1, top-p 0.95, top-k 20.
- Thinking enabled, medium reasoning effort, **4,096 thinking tokens** per
  answer, at most 16,384 total generated tokens per answer.
- Previous assistant reasoning, code and tool output remain in the history.
- Up to 32 model requests per stage and 40 minutes per engine, exclusive C1.
- A fresh serving worker and excluded compiler warmup before each run.
  Prefix caching stays enabled within the session.
- Tool results normalize incidental test timings and absolute host paths.
  Raw outputs and streamed responses are archived separately.

The thinking budget is a client benchmark setting, not a production-profile
change. A fixed seed does not guarantee identical adaptive histories across
different checkpoints, engines, versions or repeated runs. The comparison
holds the task and client policy fixed, then measures the histories produced.

## Tests and measurement

The runner exposes list/read/write/exact-span-edit, visible Node tests and
software-rendered browser tests. Independent final acceptance checks stay
outside the writable agent project and are not sent back to the agent. Scores
count functional cases, not the number of self-authored tests.

Node code runs in Bubblewrap with an isolated filesystem/network and no host
home or GPU access. Independent grading mounts the project read-only. Browser
tests use pinned Chromium with SwiftShader, block external runtime requests,
and close before another model request. They cover actual WebGL2 graphics,
context loss, controls, game states, responsive layout and the real entry point.

Every request has native Prometheus snapshots before/after, with complete
accounting validation. Context bands are `[0,10000)`, `[10000,20000)`, etc.,
using full rendered input tokens including retained reasoning and tool history.
Rates are weighted:

- Prefill = newly computed KV tokens / native prefill seconds. Cached tokens
  are reported separately and never counted as new prefill work.
- Decode = generated tokens after the first / native decode seconds. Generated
  reasoning counts; the first token belongs to prefill.
- End-to-end time includes tools, metric polling and fixed grading between
  stages. Worker startup and the initial warmup are excluded.

The context target is at least 100K, preferably roughly 130–150K. It is a
calibration/coverage target, not a prompt instructing the agent to generate more
history. Empty bands stay empty; request counts and actual maximum context must
be published. If the first full GPTQ run is below 100K, the campaign stops before
EXL3 so that an expanded fixture can be authored and frozen as a new version.

## Run on the local B70 installation

Measured tooling: Node 24.20.0, Playwright 1.63.0 and Chromium Headless Shell
153.0.8010.12. Python, Docker, user systemd and Bubblewrap are also required.
Install the pinned browser harness from the repository root:

```bash
npm ci --prefix benchmarks/web-coding-harness
npm exec --prefix benchmarks/web-coding-harness -- playwright install chromium --only-shell

python3 scripts/run-web-coding-campaign.py \
  --fixture benchmarks/web-coding-fixture/v7 \
  --output benchmark-results/flappybird-new-run --max-wall-seconds 2400
```

The campaign acquires the shared benchmark lock, runs GPTQ then EXL3 on the same
B70 at 180 W, and restores the pinned GPTQ production service in `finally`.
EXL3 uses the locally installed 4.00-bpw checkpoint and image, not a new download.
`EXL3_MODEL_DIR` can locate that same checkpoint elsewhere. When launched as a
systemd job, use `--restore-production` as an additional `ExecStopPost` hook.

For another compatible OpenAI/vLLM endpoint, start its engine yourself and use
the standalone runner with a fresh output directory:

```bash
python3 scripts/run-web-coding-benchmark.py \
  --fixture benchmarks/web-coding-fixture/v7 \
  --base http://127.0.0.1:8082 --container b70-exl3xpu-coding \
  --output benchmark-results/flappybird-other-engine
```

The endpoint must expose matching native phase counters; client TTFT is not
silently substituted for native prefill time. Model name is discovered from
`/v1/models`. Compare results only when fixture, runner, sampling, test harness
and limits match. Use the summary exporter to validate and publish records:

```bash
python3 scripts/grade-web-coding-output.py benchmark-results/flappybird-new-run/gptq
python3 scripts/grade-web-coding-output.py benchmark-results/flappybird-new-run/exl3
python3 scripts/summarize-web-coding-benchmark.py \
  --gptq benchmark-results/flappybird-new-run/gptq \
  --exl3 benchmark-results/flappybird-new-run/exl3 \
  --output benchmarks/runs/flappybird-new-run
```

This compares two complete serving recipes and quantized checkpoints. It does
not isolate quantization from vLLM version, MTP depth, draft vocabulary or other
profile differences. Functional scores describe this particular task, not a
general coding-quality ranking.

## Calibration and request budgeting

Version 1 exposed a browser-harness context-restoration mistake. Version 2
corrected it and reached the last audit with a working game, but requested
16,384 output tokens on top of 190,084 input tokens and was rejected. Version 3
added exact token preflight and a focused final guide/check stage. Its 2K-budget
agent nevertheless spent the physics phase repeatedly correcting self-written
tests: the 32-request phase limit was reached despite 22/22 independent cases
passing, then the same test failures carried into controls. These are partial
calibration attempts, excluded from headline comparison.

Version 4 keeps the same small game, starter, exact APIs and 45 independent
acceptance cases. It returns to a common 4,096-token thinking budget and removes
mandatory self-test development. Supplied visible contract checks and the real
browser are primary; optional pure-module regressions are at most three small
cases per module. Large DOM/WebGL mocks are explicitly outside this first task.
Both scored engines start from the untouched v4 starter. No history compaction
or context-padding requirement is introduced.

Before inference, the runner calls `/tokenize` with exact retained history and
tools, normalizing the reasoning alias as ChatCompletionRequest does. It
reserves at most 16,384 output tokens within a common 200,704-token window (or a
smaller engine limit) and checks each preflight count against inference usage.
CPU preflight overhead is included in end-to-end time and recorded separately
from native prefill/decode phases. See the calibration ledger for exact
excluded-attempt details.

Version 7 corrects one storage-fault harness assumption: eager and lazy adapters
are both accepted, provided a real access reports the error. The task, starter,
sampler and other 53 cases are unchanged. The overnight timed pair used v6;
its raw scores and supplemental contract review remain separate. V7 is the
corrected fixture for future repeats and was not timed overnight.

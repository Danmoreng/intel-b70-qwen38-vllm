# Web coding harness

The published overnight benchmark used the frozen
[fixture v6](../web-coding-fixture/v6/README.md) and the versioned `v3` checks.
Its manifest pins those four harness files and the coding-loop source before
either engine starts. Use the fixture's run command from the repository root,
then the final-output grading/export sequence below.
`npm ci --prefix benchmarks/web-coding-harness` installs the pinned
Playwright dependency shared by these ES-module scripts.

The final independent score contains **37 Node cases and 17 browser cases**.
The browser's renderer-identity row is metadata, excluded from the denominator.
`checks.mjs --acceptance` includes sealed edge cases; the model's test tool does
not receive those cases. Browser tests are visible to the model. They use real
WebGL2 through CPU SwiftShader and block external runtime requests. The Node
checks run with a read-only project in Bubblewrap.

The unversioned scripts belong to the initial pilot. Version 1 was stopped
because its context-loss harness re-queried an extension while the context was
lost; version 2 retains the extension before loss and restores through it.
Fixture v3 adds exact context-budget preflight and a focused final audit.
Fixture v4 removes mandatory self-test suites and large DOM/WebGL mocks while
retaining the same game and its 45 original acceptance cases. Both engines use 4K
thinking budgets throughout fixture v4–v7. The earlier abandoned attempts are documented in the
[calibration record](../web-coding-calibration-20261001.md), excluded from
headline results.

## Session outcomes and final-output grading

The shared input/output window is 200,704 tokens. Exhausting it is an incomplete
task outcome, not an invalid throughput sample or a faster completion. The
campaign proceeds to the other engine after this specific failure; unrelated
infrastructure failures still abort. The 32-request cap per stage is also
reported explicitly when the agent does not declare that phase complete.

After both timed sessions, grade each unmodified project with the same frozen
54 cases, then export the comparison. This also grades projects from a stopped
session. Post-run grading time is excluded from session time and native rates.

```bash
python3 scripts/grade-web-coding-output.py benchmark-results/flappybird-new-run/gptq
python3 scripts/grade-web-coding-output.py benchmark-results/flappybird-new-run/exl3
python3 scripts/summarize-web-coding-benchmark.py \
  --gptq benchmark-results/flappybird-new-run/gptq \
  --exl3 benchmark-results/flappybird-new-run/exl3 \
  --output benchmarks/runs/flappybird-new-run
```

## Supplemental visual review

V5 supplied common coordinate guidance but completed at only 83K. V6 adds
settings and persistent local highscores with nine additional cases, following
the user’s conditional request for a longer useful task. Use
`--max-wall-seconds 2400` to reproduce its equal 40-minute per-engine budget.

Run the same postflight on each generated project **after** inference:

```bash
node benchmarks/web-coding-harness/postflight.mjs \
  benchmark-results/flappybird-new-run/gptq/project \
  benchmark-results/flappybird-new-run/gptq-visual-review
node benchmarks/web-coding-harness/postflight.mjs \
  benchmark-results/flappybird-new-run/exl3/project \
  benchmark-results/flappybird-new-run/exl3-visual-review
```

It compares pixels against an otherwise identical scene without the bird and
pipes: both pipe bodies must be visible, the gap must remain open, and the bird
must be visible. It then drives the actual mounted app through 1,500 manual ticks
using a fixed gap-following controller, captures its frame-450 scene and checks
that play continues with at least three points. It writes JSON plus screenshots
without editing the graded code. These are supplementary observations, not
extra cases added to the frozen 54-case score or inference timings.

An optional standalone figure uses the validated exported metrics:

```bash
uv run --with matplotlib==3.11.2 python3 scripts/plot-web-coding-benchmark.py \
  benchmarks/runs/flappybird-new-run/comparison.json
```

The figure preserves empty bands as gaps; no interpolation supplies unmeasured
results. Request counts and exact numeric records remain in the exported report.

## Corrected storage-access contract

The overnight v6 measurements used harness v3 unchanged. Its storage-error
probe incorrectly required eager reading in the factory. Post-run contract
review accepts eager or lazy access, reuses identical browser cases and grades
unchanged sources. Raw scores and all timed requests remain available.
[Harness v4](v4/checks.mjs) corrects only this probe; [fixture v7](../web-coding-fixture/v7/README.md)
is the future repeatable task, not an overnight measurement.

```bash
python3 scripts/review-web-coding-contract.py benchmark-results/flappybird-new-run/gptq
python3 scripts/review-web-coding-contract.py benchmark-results/flappybird-new-run/exl3
```

The separate fixed-prompt throughput control is not a gameplay replay system.
It reuses archived coding histories with 1,024 output tokens and cold KV:

```bash
python3 scripts/run-web-coding-replay.py \
  --source benchmark-results/web-coding-20261001/comparison-v4/gptq \
  --output benchmark-results/flappybird-control
python3 scripts/summarize-web-coding-replay.py \
  benchmark-results/flappybird-control benchmarks/runs/flappybird-control
```

The strict three-point control encountered an EXL3 preemption at 139K even on a
fresh worker. The [cold comparison](../runs/2026-10-01-flappybird/COLD_COMPARISON.md)
therefore distinguishes zero-preemption controls from exact-count cold
**preempted diagnostics**; all attempts and native counters are retained.
`run-web-coding-cold-diagnostic.py` records a separate fresh-worker request with
preemptions reported explicitly. It does not tune or replace the EXL3 image.

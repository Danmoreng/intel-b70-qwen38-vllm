# Flappy Bird benchmark calibration — 1 October 2026

The user requested a small first attempt: a dependency-free WebGL2 game with
seeded deterministic physics, collisions, scoring, keyboard/touch input and
accessible responsive start/pause/restart controls. There is no level editor or
replay system. An oversized draft was removed before any model saw it.

The observation target is at least 100,000 natural input tokens, preferably
130,000–150,000. The prompt never asks for padding or a minimum runtime. The
small task already exceeded 100K during calibration, so no larger application
was added. Duration and context growth are outcomes.

## Excluded calibration attempts

These attempts are not headline benchmark samples. Raw requests, responses,
original runner sources and generated projects remain under
`benchmark-results/web-coding-20261001/`. Changes below were made before either
engine ran the eventual qualifying frozen protocol; EXL3 was not used to select
or tune the task. Generated code was never repaired by the host.

| Attempt | Outcome and resulting protocol correction |
|---|---|
| `pilot-gptq` | Stopped in foundations: the runner omitted previous assistant reasoning and left effort partly implicit. Retain reasoning and fix effort explicitly. |
| `small/gptq` | The corrected 8K-thinking pilot reached 160,176 input tokens during controls and was stopped before overflow. Mandatory whole-file readbacks were removed; an exact unique-span edit tool and clearer input examples were added. |
| `comparison-v1/gptq` | Stopped during graphics after 62 requests at 152,480 input tokens. A genuine one-color renderer failure also exposed a separate browser-harness context-restoration bug. Correct the harness lifecycle before a fresh run. |
| `comparison-v2/gptq` | Completed five stages with all 32 cumulative Node cases and stage-5 browser cases passing. Request 91 failed because 190,084 input plus the requested 16,384 output exceeded the 200,704 window. Add exact token preflight/output reservation and a focused final guide/check phase. |
| `comparison-v3/gptq` | The 2K-thinking candidate hit the 32-request physics cap despite passing all 22 independent cases, then spent controls repairing false self-authored assertions. Stopped around 150K. Remove mandatory broad self-test development and return to 4K thinking. |

A failed candidate is not a completed game or evidence of an engine-kernel
failure. Calibration results are retained rather than combined with the final
performance table. Earlier fixture versions pin their historical runner hashes;
they require those archived runner sources, not the current runner.

### Measurement and contract corrections

The first-three-stage empty starter fails all 31 acceptance cases. Calibration
caught an `assert.throws` check that could accept a missing export; required
functions are now verified before invalid-input checks. Public contracts also
clarified two cases before comparison: ignoring inherited configuration values
is valid, and an empty input-binding array is not forbidden merely because its
members must be nonempty strings.

The v1 browser harness queried `WEBGL_lose_context` after context loss, when
WebGL returns null. An independent empty-canvas probe confirmed this. Harness
v2 caches the extension before loss and restores via that reference. On the
same unmodified generated renderer, this correction passed restoration while
still rejecting its one-color output. The final 45-case harness is v2.

The capacity diagnostic used a one-token, unscored request to verify 190,084
actual input tokens. Passing historical `reasoning_content` directly to
`/tokenize` instead returned 112,255 because this endpoint did not apply the
ChatCompletionRequest alias normalization. Normalizing to `reasoning` gives the
correct 190,084 count. Every measured request now preflights that exact template
and verifies the subsequent inference count. Its output cap is the smaller of
16,384 and the remaining shared 200,704-token input/output window. No history
is compacted or padded.

Eight runner regression checks cover weighted native phase rates, exclusion of
the first generated token, delayed counter settlement, competing-traffic
rejection, streamed reasoning/tool assembly, normalized tool history, atomic
unique-span editing, and context preflight/alias/output reservation. Browser
checks use Playwright 1.63.0 and Chromium Headless Shell 153.0.8010.12 with
SwiftShader, so browser execution does not compete for the B70 inference GPU.

## Initial retained stress protocol: v4

`flappybird-webgl2-v4` keeps the same small game and corrected 45-case harness.
Its six stages cover foundations, physics, controls, WebGL rendering, the
integrated game and a focused release guide/check. Supplied contract checks
and the actual browser are primary. Optional pure-module regressions are at
most three small cases per module; large DOM/WebGL mocks are forbidden.

Both engines use medium reasoning effort, a 4,096-token thinking budget,
16,384 nominal output cap, temperature 1, top-p 0.95, top-k 20 and seeds
73,000 + request index − 1. They receive untouched starter copies and fresh
workers. Fixed sampling seeds do not guarantee identical adaptive histories.

Fixture manifest SHA-256:
`58b09f4db9b6c3c702156294c54aa206a2315a328914add92f297bf2bbb8e331`.
Runner SHA-256:
`a520cd59ac86c3e8631883dacb85c824ce467aed21e9964d70c27cc572d98e82`.
The GPTQ campaign started at 02:21 UTC. Stage grading is archived and not
returned to the agent; visible contract/browser results are available.

GPTQ exhausted the shared context window after 98 valid measured requests in
66.5 minutes. Its final measured input was 200,509 tokens; the following
preflight counted 200,705 and stopped before inference. Three stages were
declared complete, graphics hit its request cap, and game integration began;
release was never reached. This is the retained v4 task outcome, not another
excluded calibration. The assignment and sampler were not changed to obtain
a successful GPTQ game. An EXL3-only campaign was started at 03:30 UTC with the
same frozen v4 runner, fixture, limits and untouched starter.

Both valid measured trajectories are eligible regardless of task completion.
A context-budget failure is labelled as an incomplete task, and its elapsed
time is never treated as a faster completion. Final output from each engine is
graded unmodified after its session using the same frozen 45 cases, with those
post-run grading seconds excluded from the timed session. Infrastructure or
native-accounting failures remain invalid. The campaign wrapper now proceeds
to the second engine after this specific task-budget failure; it continues to
abort on unrelated failures. The actual wrapper source used by each overnight
campaign is archived locally.

## EXL3 identity and interpretation

The user confirmed the locally installed **4.00-bpw** checkpoint:
`turboderp/Qwen3.8-27B-exl3`, revision
`113cf7ab958054860e43fb7f3063b1af19171095`, with a 6-bpw target output head and
4-bpw MTP weights. Both shard SHA-256 hashes match cached download metadata.
Image:
`sha256:cba73584f4ab0a2b37eac1356f34f16655ac5e740d845e110997b03be78279b7`.

The tokenizer file hashes and some preprocessing settings differ from GPTQ,
although vocabularies, normalized merges and chat-template bytes match. Each
engine's actual rendered token count determines its context band. The
comparison covers complete recipes, including different vLLM versions, MTP
settings, draft vocabularies and quantization/head formats. One fixed-seed
adaptive trajectory cannot isolate quantization effects or establish a general
code-quality ranking.

## Supplemental output review

`postflight.mjs` was prepared during inference and runs after both timed
sessions. It checks the same canonical running scene for bird pixels, both pipe
bodies and an open gap, then plays 1,500 manual app ticks with a fixed
pipe-gap-following controller and captures frame 450. Successful play means
remaining running with at least three points. These observations supplement
the ready-state renderer case, which starts without pipes; they do not change
the 45-case denominator or inference timings. Both generated projects remain
unmodified.

## Retained v4 outcome and second variant

The initial v4 pair remains public under
[runs/2026-10-01-flappybird-stress-v4](runs/2026-10-01-flappybird-stress-v4/README.md),
with full numeric records, compressed conversations, projects and supplementary
visual evidence. Final read-only grading finds **35/45 GPTQ** and **44/45 EXL3**
cases. EXL3 used 116 requests, 49.1 minutes and reached 141,551 input tokens;
five stages were declared complete, with graphics capped. Its controller
survives 1,500 ticks and scores 12, but bird and pipes are not visible. GPTQ
exhausted context at 66.5 minutes and its final app has a syntax error. Neither
output is a usable rendered game. Uniform case scores do not express severity.

Version 5 is a second exploratory benchmark variant, not a repair of either
measured project. It changes only the stage-4 instruction, providing both
profiles the same textbook pixel-to-clip formulas, explicit width/height and
vertex-stride/count reminders, and asking them to correct their geometry rather
than patch browser prototypes. The entire starter, other five instructions,
4K thinking budget and all 45 acceptance cases remain identical. Both profiles
start new workers, untouched starters and conversations. Its per-engine wall
budget is 3,300 seconds (55 minutes), equal on both sides.

V5 manifest SHA-256:
`ab332157914bd4dacdbd259c36a225f8f0835427485cc02c45fd7701afae27df`.
The coding runner remains
`a520cd59ac86c3e8631883dacb85c824ce467aed21e9964d70c27cc572d98e82`.
The fresh v5 campaign started at **04:24:20 UTC**. Earlier stress results remain
separate and visible. Task selection used exploratory observations on both
profiles, so this is a reproducible selected workload, not an unbiased estimate
of general coding quality across arbitrary tasks.

A separate three-prompt cold-KV control is prepared using archived v4 histories
nearest 100K, 140K and 190K. Both profiles receive identical messages, tools,
sampling and template settings, with only the served model ID differing. Each
uses a distinct cache salt and exactly 1,024 generated tokens with ignored EOS;
zero cached tokens and exact native counters must be verified. Tool calls are
not executed and truncated outputs receive no coding-quality score. This
control separates matched input throughput from adaptive trajectory length.

## Successful short v5 and conditional extension to v6

GPTQ completed v5 on 1 October at 04:45 UTC in **20.0 minutes**, with **57
requests, 83,146 maximum input tokens, all six stages declared complete and
45/45 final functional cases passed**. No EXL3 v5 sample was started: the
campaign's coverage gate stopped because a successful complete task was below
100K. This is a successful but short calibration, not a paired comparison.
The unchanged early task stages also followed a shorter trajectory than v4;
the sampler seed is fixed, but adaptive repeat lengths are not guaranteed.

The user explicitly allowed a longer task only if the first useful attempt was
too short. Version 6 therefore adds a final-stage settings panel and locally
persisted highscore list, preserving the small game and geometry reference.
It adds five independent score-store contracts and four actual browser cases,
for **37 Node + 17 browser = 54 functional cases**. There is still no editor,
replay, backend or third-party runtime dependency. The first five stage
instructions are unchanged. New protected public specs describe atomic config
validation/reset, editable-field keyboard isolation, stable top-ten ordering,
detached data, invalid/corrupt storage, one record per gameover and reload/clear.

Harness v3 preserves the original 45 cases and adds these nine. Its default
browser action timeout is three seconds, so absent new controls fail a case
without turning an incomplete game into a test-infrastructure timeout. A copied
successful v5 game plus an independent temporary score-store reference passes
all **37 Node** cases; all **13 original browser** cases pass and exactly the
four absent extension cases fail. The measured v5 tree was not edited. Raw
calibration proof is archived as `harness-v3-calibration.json` locally.

V6 uses the same frozen coding runner, 4K thinking and seed policy. Its identical
per-engine wall budget is **2,400 seconds (40 minutes)**, chosen to fit the
remaining overnight window. A context or wall-budget stop is published as an
incomplete task, never a faster completion. Fixture manifest SHA-256:
`a201091d9eb5f81c689c47a6e6293b2d2bf910ce9f68653d35ad217fdc82c695`.
The fresh paired campaign started at **04:58 UTC**. This is the selected expanded
workload; v4 stress failures and v5 short success remain separate evidence.

## Final paired v6 results and contract correction

The paired campaign finished on 1 October at 06:09 UTC. GPTQ recorded 107 valid
requests, reached 152,762 context tokens and stopped at its 40-minute task budget
(40 min 19 s actual, including its final in-flight request). It declared four
of six stages complete. EXL3 completed all six stages in 28 min 8 s, with 87
requests and 106,434 maximum context tokens. Both retain full reasoning/history;
no padding, compaction or generated-source repairs were introduced.

GPTQ passes 51/54 frozen final cases and EXL3 54/54. Source review found one
harness mistake: storage errors were required already during factory creation,
although lazy access is permitted by the public API. The corrected common
post-run probe accepts construction or actual-access errors; unchanged GPTQ
then passes 52/54 and EXL3 stays 54/54. Both pass external schema and read/write
error probes. GPTQ's remaining two browser cases stop at a missing name-field
test attribute; the labelled input exists. This cannot establish the remaining
case behaviors, and it does not establish that persistence is absent.

Harness v4 changes only the faulty Node probe. Fixture v7 keeps v6's exact task
instructions and starter with that corrected harness; it is prepared for future
runs and **was not measured overnight**. V6's timings and feedback retain the
overstrict case, so its task-time difference is not solely a model-quality
comparison. The quality review and raw/contract-reviewed JSON retain both scores.

Both generated games pass the supplemental real-browser rendering and
1,500-tick/12-point play probe. A roughly 4.6-second CPU SwiftShader GPTQ review
overlapped EXL3 requests 4–5; exact timestamps are archived. B70 inference stayed
exclusive. Matched-prompt control follows all reviews and uses fresh serving
workers, distinct cache salts and zero cached tokens on both profiles.

## Cold-control preemption finding

The initial three-point strict zero-preemption control completed all GPTQ
points and EXL3 at 102,752 tokens. EXL3 at 139,193 tokens completed 1,024 outputs
but failed the zero-preemption criterion. A repeat on a fresh EXL3 worker
confirmed **one preemption**, zero cache hits and exact input/output/native
counts. This is a resource/scheduler observation of the pinned EXL3 profile,
not a changed image or an API token-budget failure. The failed strict attempts
remain archived. The matched cold report separates clean controls from
preempted diagnostics, and does not silently replace a failed clean control.
A separate fresh-worker 187,695-token diagnostic is measured last. Native
throughput observations with preemptions must retain that qualification.

The final fresh-worker 187,695-token EXL3 diagnostic completed with exact native
counts, zero cache hits and **one preemption**. All three matched input points
are exported with explicit preemption classifications. Cold prefill differs by
less than 3% in these observations; GPTQ decode is higher at all three. They are
single samples with truncated outputs and do not score code quality. The
production restore begins immediately after the last diagnostic completes.

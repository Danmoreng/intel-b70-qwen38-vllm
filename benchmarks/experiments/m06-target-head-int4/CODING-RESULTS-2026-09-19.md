# M06: real-agent coding comparison, 2026-09-19

The M04 control completed both tasks and passed all nine independent acceptance
checks. The INT4 target-head candidate completed the configuration task but
exhausted the common 100,000-output-token budget during the status-display
follow-up. It was stopped after 53:49, not by the 60-minute time limit.
No profile or generated dashboard change was promoted.

## Task outcome

| Metric | M04 control | INT4 target head |
|---|---:|---:|
| Agent wall time | 42:47, completed | 53:49, output-budget stop |
| First task completed | 18:37 | 44:30 |
| First-task generated tokens | 44,543 | 87,626 |
| Second task | Completed | Incomplete |
| Successful model requests | 133 | 100 |
| Agent tool calls | 131 | 119 |
| Generated tokens | 79,308 | 100,000 |
| Maximum measured prompt context | 143,034 | 172,528 |
| Independent acceptance | 9/9 | 8/9 |
| Repository tests | 570 passed, 6 skipped | 540 passed, 6 skipped |
| Build, type checks, lint | Passed | Passed |
| Formatting | Only seven pre-existing warnings | Same seven pre-existing warnings |

Different repository test counts reflect tests added by each agent; the same nine
external checks are the comparable acceptance criterion. Both generated fixes
pass the eight configuration checks. The candidate did not implement the status
follow-up, so its remaining safety-check wording fails the ninth check. This is
an unfinished task, not a demonstrated defect in its completed configuration fix.

The six skipped tests require host-only nested sandbox execution; the existing
B70_RELEASE_BUILD flag excludes them in both isolated workspaces. Tests ran with
two workers. Validation was performed after both arms, independently of the
agents' own claims. No full manual production-readiness review is implied.

## Whole-session native metrics

| Metric | M04 control | INT4 target head |
|---|---:|---:|
| Prefill compute | 637.49 tok/s | 563.17 tok/s |
| Decode after first token | 54.39 tok/s | 55.82 tok/s |
| Native prefill time | 992.77 s | 892.60 s |
| Native decode time | 1,455.66 s | 1,789.58 s |
| Prefix-cache hit rate | 93.75% | 94.64% |
| MTP accepted / proposed | 52.20% | 52.06% |

The aggregate decode difference is +2.63%, but these means cover different
context distributions, generated content and completion outcomes. The aggregate
prefill difference (-11.66%) is likewise not evidence of a kernel regression.
Tool execution is excluded from the native phase rates. Prefix-cache hits are
reported separately from newly computed prompt tokens.

## Rates by logical prompt context

Weighted ratios of token sums to native phase-time sums; K here means 1,000 tokens.
Every model request has a validated native counter pair. Sample counts vary,
and these are adaptive trajectories, not matched prompt/output replays.

| Context | n M04 / INT4 | Prefill M04 | Prefill INT4 | Decode M04 | Decode INT4 | Decode change | MTP M04 / INT4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–10K | 15 / 4 | 1,427.61 | 1,489.66 | 67.87 | 81.79 | +20.50% | 49.6% / 57.9% |
| 10–20K | 6 / 5 | 1,232.84 | 1,269.02 | 55.39 | 57.92 | +4.56% | 38.3% / 37.5% |
| 20–30K | 6 / 2 | 1,066.85 | 1,032.50 | 53.14 | 61.08 | +14.93% | 37.3% / 43.0% |
| 30–40K | 5 / 3 | 953.41 | 931.99 | 56.65 | 71.77 | +26.68% | 44.7% / 57.3% |
| 40–50K | 8 / 3 | 822.13 | 851.36 | 75.06 | 76.42 | +1.82% | 71.2% / 67.0% |
| 50–60K | 3 / 5 | 766.68 | 759.55 | 68.28 | 70.11 | +2.69% | 65.3% / 62.7% |
| 60–70K | 20 / 7 | 697.40 | 707.65 | 61.74 | 59.20 | -4.12% | 60.4% / 51.9% |
| 70–80K | 14 / 8 | 642.16 | 644.11 | 48.84 | 55.56 | +13.77% | 44.4% / 50.1% |
| 80–90K | 5 / 5 | 606.94 | 602.38 | 43.55 | 55.18 | +26.68% | 39.4% / 51.4% |
| 90–100K | 7 / 8 | 561.06 | 554.79 | 42.89 | 47.25 | +10.17% | 40.3% / 43.0% |
| 100–110K | 5 / 8 | 516.94 | 518.09 | 41.56 | 61.86 | +48.86% | 40.8% / 66.9% |
| 110–120K | 6 / 11 | 485.07 | 483.45 | 44.64 | 56.20 | +25.90% | 47.1% / 61.9% |
| 120–130K | 11 / 13 | 460.55 | 461.00 | 52.34 | 42.19 | -19.40% | 62.3% / 42.5% |
| 130–140K | 18 / 5 | 434.11 | 446.53 | 55.15 | 48.51 | -12.04% | 69.8% / 53.7% |
| 140–150K | 4 / 5 | 423.86 | 418.76 | 41.44 | 43.26 | +4.39% | 47.7% / 48.6% |
| 150–160K | — / 3 | — | 399.45 | — | 39.22 | — | — / 42.4% |
| 160–170K | — / 3 | — | 381.33 | — | 35.51 | — | — / 38.3% |
| 170–180K | — / 2 | — | 367.42 | — | 36.98 | — | — / 42.3% |

Prefill is close within common bands (roughly within 4.4%). Decode gains change
sign and size along with MTP acceptance: at 100–110K, INT4 reaches 61.86 versus
41.56 tok/s while acceptance is 66.9% versus 40.8%; at 120–130K the direction
reverses, with 42.19 versus 52.34 tok/s and 42.5% versus 62.3% acceptance.
This is consistent with a strong workload/trajectory contribution. This single
pair cannot establish that the quantized head causes worse task quality, nor
can the best individual band establish a pure INT4 speedup.

## Card energy

Both runs stayed at the verified 180 W cap. These are card energy-counter
differences over the entire agent session, including tool/idle time; they are
not wall-socket system energy or decode-only power.

| Metric | M04 control | INT4 target head |
|---|---:|---:|
| Card energy | 123.99 Wh | 139.57 Wh |
| Mean card power | 173.85 W | 155.62 W |
| Generated tokens/J | 0.1777 | 0.1990 |

The candidate produced more tokens per joule (+12.01%), but consumed 12.57%
more total card energy while failing to finish both tasks. Lower average power
also includes much more tool time (536.23 versus 104.38 summed tool seconds);
it is not by itself proof of a more efficient inference kernel.

## End condition and verification

The candidate reached exactly 100,000 generated tokens in 100 successful model
requests, below the 160-request and 60-minute ceilings. Its final response was
limited to the remaining 2,552-token budget and ended with finish_reason=length.
A subsequent compaction/recovery attempt received HTTP 429 request_or_output_budget
from the benchmark relay. No successful compaction followed. All completed
inference requests still have valid usage and native timing records.

The control settled after each of its two tasks. The candidate settled normally
after the first task; its final settled event follows the budget failure and
must not be mistaken for successful task completion.

All 233 successful-request counter pairs were independently recomputed from the
saved raw Prometheus snapshots and agree with the recorded token and phase-time
values. Both arms reported no telemetry collection errors. The first warm-up-only
infrastructure attempt (missing dev cache-reset endpoint) is retained separately
and contributes no scored requests. Both scored arms enabled the same loopback
development cache endpoint and verified cache-reset success after warm-up.

## Conclusion

Keep the qualified M04 production profile for now. The earlier fixed-context
screen supports an INT4 decode-speed opportunity; this real-agent pair does not
demonstrate improved completed-task performance. A stronger follow-up would use
repeated paired tasks with the same larger output budget for both profiles and
separate fixed-request replay to isolate inference speed. Do not extend only the
failed arm and present that as the original matched-budget result.

Production was restored to the pinned M04 image at 180 W and HTTP 200. A fresh
health/image check during this analysis confirmed the restoration. Generated
dashboard changes remain in isolated workspaces; the live Tasks configuration
has not been changed by this benchmark.

Machine-readable numbers: [coding-results-20260919.json](coding-results-20260919.json).
Local detailed evidence: [run directory](runs/coding-20260919-101848/).
Task, harness and limitations: [coding/README.md](coding/README.md).

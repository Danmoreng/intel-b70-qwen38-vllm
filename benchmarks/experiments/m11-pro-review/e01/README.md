# E01 — opt-in XPU QK/RoPE/gate preparation fusion

Status: **not promoted**. Operator and small quality gates passed, but the
five-pair serving comparison found no reliable gain. See [the decision](decision.md). The separate image is based on the unchanged production
Runtime 26.35 / IGC 2.41.5 image.

`patch.py` enables the existing Triton implementation only with
`B70_FUSED_QK_ROPE_GATE=1`, XPU, TP=1, 24/4 heads, head dimension 256,
rotary dimension 64 and FP16. Existing CUDA and model eligibility are retained.
Per-call guards inspect metadata only and accept normal split-view row strides.
Unsupported metadata returns to the existing route before launch. Unexpected
kernel failures are not hidden with a fallback exception handler. The gate
output is the raw gate copy; sigmoid remains in the model's original forward.

## First operator screen

`LATEST-SCREEN` points to the local evidence. Real norm weights from target
layers 3/35/63 and the MTP layer; seeded synthetic QKV inputs. 180 cases cover
M=1/2/3/4/5/16/64/256/1663/1664/1665/3328/4096/4992/6656 and text, high text,
and divergent synthetic mRoPE positions. All numerical cases passed. Graph
replays also passed. Eager and graph samples are retained separately; the
minimum graph-region latency reduction was 68.14% in this screen.

## Actual projection replay

`LATEST-REAL` points to 26 real activation cases captured from the unchanged
projection path, with a diagnostic-only eager serving image. Capture was enabled
only after startup and was disabled before timings. Target layers 3/35/63 and
the actual MTP layer are represented, with text and a real image request.
Measured row counts were 1, 5, 992, 1664, 1785 and 3328. The vision data includes
different T/H/W position rows. Norm weights and the actual rotary cache were
saved with the projections; captured artifacts stay local in ignored `runs/`.

All 26 cases and their graph replays passed. Gate and V outputs were bitwise
equal. Q/K passed the predeclared `atol=0.01, rtol=0.002`; the largest absolute
difference was 0.015625, which passes the combined absolute/relative rule at
the corresponding values. The complete preparation region's captured-graph
latency fell by 58.45–85.90% across these cases, using 21 alternating blocks.

These operator comparisons exercise installed norm/rotary calls captured into
XPU graphs. They are not measurements of the complete torch.compile model,
full attention/logits or end-to-end tokens/s. The paired serving study must
establish that benefit separately; no operator percentage is an advertised
serving gain.

## Serving gate

Five paired blocks, alternating control/candidate order, with frozen exact-token
8192/16384 requests. Each arm uses the same patched image with its flag off/on,
production compile/graphs, MTP4, Q128/M04, 4096 scheduled tokens, 200704 context
and 180 W. Cold requests reset cache; warm requests prime the same prefix before
the identical full request. Both shapes warm up with 256 output tokens.

The harness waits for finished-request counters and uses the native computed
prefill token histogram rather than all prompt tokens. It records preemptions
and excludes any preempted case from this paired screen. Phase times, TTFT,
actual computed tokens, cache hits and MTP acceptance are kept together.
Uninstrumented timings are separate from the one short profiler request.
Coding assertions, teacher-forced NLL, real vision/tool continuation and prefix
checks follow the first block. A production-sampling request is separate from
the deterministic throughput track.

A clear paired serving win must precede 32K/64K, full-context and long coding
qualification. The old production service is restored after each standalone
study and by systemd recovery if a study fails. E02 is explicitly skipped by
the user's later instruction; this work does not change the prefill budget.

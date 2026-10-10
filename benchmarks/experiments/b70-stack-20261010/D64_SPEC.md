# Bounded 64K regression diagnostic

First reuse the qualified v0.31 performance-review-measurements, cross-version
quality and matched-state replay results. Old-image full qualification is not
authorized by this task. Existing old/new text trajectories differ while
v0.31 repeats match; same-window round-equivalent median is nearly unchanged.
That does not establish a causal acceptance explanation or GPU-only latency.

If saved state/logits do not answer a named question, collect only the first
same-token-prefix divergence and one later long-context window. Restore KV,
GDN/convolution state, request slots, draft inputs and RNG. Compare actual
computed target and all three MTP proposals before any frozen-output copy,
configured sampler transforms and relevant acceptance decisions. Cross-version
states must be reconstructed using each compatible runner; do not transplant
graph objects. Record separately distribution change, iteration efficiency
and any falsifiable state error. A narrow old-worker run requires an explicit
missing-reference justification in its receipt. No unbounded bisect.

Stop after accepted-tolerance trajectory differences are supported or one
localized reference failure is isolated. Do not tune sampling or MTP depth.

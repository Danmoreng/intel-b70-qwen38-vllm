# D64 — bounded same-prefix diagnostic

Decision: no localized regression is identified to patch. Stop the cross-version diagnostic at its preregistered budget. Root-cause attribution of the historical throughput/acceptance difference remains **inconclusive**; no serving or sampling change is made.

The archived 64K streams lack exact emitted IDs and same-prefix full-vocabulary operands. Under the plan's named missing-reference exception, capture two frozen 64K prefixes (suffix positions 0 and 32), each freshly reconstructed twice, on historical v0.30, qualified v0.31 and E01. Each request has an eight-token output budget. No old full benchmark, bisect, transplanted state or BF16 model run is performed.

Actual target logits and transformed sampler scores are copied before sampling. Actual native samples are copied before the single common anchor is teacher-forced; each runner then constructs its own MTP state. First MTP conditioning anchor is asserted in the native request state. Later results remain native. Actual proposals from all three greedy MTP steps are recorded. Their q is a Dirac distribution, not an unrecorded learned proposal probability.

All **24 observed verification rounds** agree with an independent scalar Philox probability-ratio oracle, whose generator passes three Random123 known-answer vectors. Only the accepted prefix is counted; checks after first rejection are explicitly counterfactual. Repeated captures have identical target logits, proposals and valid sampled outputs in all six version/position pairs. Uninitialized padding in one historical rejection output differs and is retained in raw data; it is excluded using authoritative `num_sampled` lengths.

E01 is **bit-identical** to qualified v0.31 for every compared target and transformed distribution on these common prefixes. The first three proposals agree. Both versions independently reconstruct their state; no frozen output replaces computed target results.

Historical v0.30 and qualified v0.31 are not numerically identical. At these selected common prefixes, full-vocabulary KL reaches 0.006319 nats, transformed probability total variation reaches 0.04484, all compared argmaxes agree and the first three proposals agree. The transformed support masks agree. This documents distribution changes capable of influencing later stochastic trajectories; it does not prove that the earlier throughput loss is entirely due to those changes, nor that every 64K distribution is equivalent. Previously reviewed short/32K upgrade quality remains the accepted scope; this bounded diagnostic does not invent a new 64K quality tolerance or reopen that completed qualification.

No stale-state, invalid acceptance-prefix, slot-repeat or candidate numerical failure is demonstrated by this diagnostic. Those finite observations are not a universal state-correctness proof. Candidate prefix reuse, abort/restart, C4/mixed graphs and admission pressure remain separate release gates.

See `D64-same-prefix-review.json` for exact capture hashes, same-conditioning row comparisons, oracle decisions and padding-aware repeatability. Raw full-vocabulary arrays and response text remain local under `benchmark-results/b70-stack-20261010/D64-bounded-r1/`.

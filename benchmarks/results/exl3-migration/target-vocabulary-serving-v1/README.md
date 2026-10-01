# Fixed-MTP3 draft-vocabulary serving screen

Measured serving effects, including changed generated sequences and acceptance work. Warm reuse is observed; no isolated kernel-speed or numeric-quality claim. Head roles share one physical module; full vocabulary removes one unique pruned head.

| Context | C | Cache | Pruned decode tok/s | Full decode tok/s | Change | Cache hits pruned / full |
|---|---:|---|---:|---:|---:|---:|
| code-4096 | 1 | cold | 70.66 | 61.60 | -12.8% | 0.0% / 0.0% |
| code-4096 | 1 | warm | 67.40 | 65.48 | -2.8% | 39.1% / 39.1% |
| code-4096 | 4 | cold | 151.91 | 139.95 | -7.9% | 0.0% / 0.0% |
| code-4096 | 4 | warm | 170.14 | 156.51 | -8.0% | 39.1% / 39.1% |
| prose-4096 | 1 | cold | 57.67 | 50.26 | -12.8% | 0.0% / 0.0% |
| prose-4096 | 1 | warm | 54.40 | 44.27 | -18.6% | 39.1% / 39.1% |
| prose-4096 | 4 | cold | 118.62 | 108.39 | -8.6% | 0.0% / 0.0% |
| prose-4096 | 4 | warm | 136.64 | 123.95 | -9.3% | 39.1% / 39.1% |
| code-49152 | 4 | cold | 15.35 | 15.25 | -0.7% | 0.0% / 0.0% |
| code-49152 | 4 | warm | 86.31 | 83.26 | -3.5% | 94.4% / 94.4% |
| code-102752 | 1 | cold | 44.03 | 43.33 | -1.6% | 0.0% / 0.0% |
| code-102752 | 1 | warm | 43.92 | 43.29 | -1.4% | 98.1% / 98.1% |

C1 reports native request-weighted decode; C4 reports aggregate client decode intervals. Cold C4 can mix decode and prefill. Each arm is fresh; single pruned-then-full order has no ABBA/order confidence.

All 60 requests complete. Full vocabulary saves252315648 bytes (~240.63MiB) of model allocation. This is one shared pruned-head allocation, not two independent copies. Actual KV capacity and output quality remain separate questions.

Decision: retain65536-row draft pruning with MTP3 for the EXL3 target profile.
Full vocabulary is slower in all12 observed cells (about1.4–18.6%), with equal
cache-hit fractions in each matched case. All60 requests complete without
observed preemption. Full vocabulary lowers model allocation by240.63MiB and
raises startup-reported KV capacity from295827 to303149 tokens (~7322 tokens),
but the measured speed tradeoff does not justify switching this target profile.
These startup capacities include hybrid-state accounting and are not16×262K.

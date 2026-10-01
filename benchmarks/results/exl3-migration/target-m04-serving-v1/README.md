# Shared-KV verification serving screen

Actual graph-path M04 dispatch proven at capture; serving includes acceptance/output variation and mixed prefill. Numerical micro gates pass separately. Final generated-route quality, mixed tails, stability and release gates remain.

| Context | C | Cache | Native decode tok/s | M04 decode tok/s | Change | Cache hits native / M04 |
|---|---:|---|---:|---:|---:|---:|
| code-4096 | 1 | cold | 70.66 | 74.49 | +5.4% | 0.0% / 0.0% |
| code-4096 | 1 | warm | 67.40 | 67.34 | -0.1% | 39.1% / 39.1% |
| code-4096 | 4 | cold | 151.91 | 154.55 | +1.7% | 0.0% / 0.0% |
| code-4096 | 4 | warm | 170.14 | 169.24 | -0.5% | 39.1% / 39.1% |
| code-49152 | 4 | cold | 15.35 | 15.51 | +1.1% | 0.0% / 0.0% |
| code-49152 | 4 | warm | 86.31 | 91.40 | +5.9% | 94.4% / 94.4% |
| code-102752 | 1 | cold | 44.03 | 50.14 | +13.9% | 0.0% / 0.0% |
| code-102752 | 1 | warm | 43.92 | 50.11 | +14.1% | 98.1% / 98.1% |

Eight fresh code C1/C4 cold/warm waves against exact reused pruned MTP3 baseline. Same512-output budgets, settings/weights/image; only separate M04 overlay + startup worker extension differ. Single pass; no ABBA/order confidence.

C1 uses native request-weighted decode; C4 uses aggregate client decode intervals. These are serving observations, not isolated kernel rates. Exact output divergence and actual cache reuse are retained. No production change follows from this screen.

All20 candidate requests complete with zero observed preemption. Both103K C1
512-token outputs are exactly identical to their native baselines; Decode rises
44.03→50.14 tok/s cold and43.92→50.11 warm (~14%). Warm49K C4 rises86.31→91.40
aggregate tok/s (~5.9%), with equal94.4% cache reuse; its output/acceptance work
still differs. Short4K effects are mixed, including−0.5% warmC4. Keep this as
an optimization candidate; do not infer universal workload gains or production
qualification from a single pass.

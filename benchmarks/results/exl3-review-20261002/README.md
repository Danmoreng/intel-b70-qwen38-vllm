# Bounded Pro-review follow-up

The [validated assessment](assessment.json) exports the targeted 2026-10-02
review campaign. It is not another full serving benchmark or BF16 reference
campaign, and does not promote the experimental image.

| Gate | Result |
|---|---|
| Host build/recovery helpers | Seven focused tests pass; reject release aliases and wrong service identity before side effects |
| Runtime source closure | All 13 active guard paths present/hash-matched; missing and changed source rejected in disposable CPU containers |
| Attention micro gates | 31 cases, 2 mutable graphs, 3 causal/future-poison cases; original rtol 0.01 / atol 0.003 |
| Cache finite soak | 81 exact lengths × 3 epochs, capacities 8 and 64; caps respected, hits/eviction observed |
| Post-warmup memory growth | RSS 978,944 / 958,464 bytes; GPU allocated/reserved growth zero |
| M04 copy correctness | 16 C1/C4 × q2–5 × short/long cases, bitwise old/new equality, caller output identity and changed-input graph replay |
| M04 targeted q4 repeat | C4 short paired mean +0.762%, 95% interval +0.678–+0.846%; C1 short -0.536% |
| Final copy decision | Direct copy only at C4; C1–C3 retain the qualified copy. Final 16-case matrix passes |
| Matched-state diagnostics | Eight actual target steps, full 248,320-row logits, 256 local attention comparisons pass |
| Exact early positions | Request 2/output 1 and request 3/output 3 captured in real mixed/C4-graph states; no winner change in these finite cases |

The native library is reused in the final candidate only after matching its
ABI, source hashes and artifact digest to the fresh first build. The cache
and route code therefore use the same native bytes as the validated soak.
Final candidate: `sha256:0e711fea1f9231a25289d812fffbde51ed93cbe7bad16c34f7fde3edf3d91737`.
Qualified production remains `sha256:c09015ce22180fbc90ef0f5070f4a7116c8d11be785067499477accf0216f21f`.

The mixed native-repeat varies even when the full 11.2 GB cache storage is
restored byte-for-byte. A small difference is visible in the first b/a
projection before GDN/attention; the pure C4 graph native-repeat is bitidentical.
Optimized attention meets the original local tolerance. Full-network logits
and token margins can still differ. This is evidence for the specific recreated
histories and layouts, not a universal determinism or coding-quality guarantee.

The first eight-round copy timing was noisy; the fixed 20-round q4 follow-up
uses 100 graph cycles/event and bootstrap resampling of ABBA round pairs.
Both passes and unfavorable timing arms are retained. These are micro-cycle
measurements, not end-to-end tokens/s. The final shorter regression matrix
checks the conditional implementation and does not replace the longer paired
timing evidence.

All raw logits, exact request histories, state/source hashes and timing samples
remain local under `benchmark-results/pro-review-20261002/`. The assessment
records each selected receipt's SHA256. The
[follow-up report](../../../docs/EXL3_PRO_REVIEW_FOLLOWUP.md) describes implementation,
reproduction and the boundary before a new runtime release can be promoted.

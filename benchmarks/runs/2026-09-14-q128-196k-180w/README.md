# 2026-09-14 Q128 196K production sweep at 180 W

This directory contains the published result records for the single-user
Q128, W4A16, MTP4/full-vocabulary production profile at a verified 180 W card
power limit. The generated prompt matrix and raw SSE captures are intentionally
excluded from Git; they contain no private corpus and are unnecessary to
reproduce the method.

## Results

| Input | Output | n | Client prefill median | Client decode median | Native prefill median | Native decode median | MTP acceptance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 128 | 5 | 1,586.23 | 100.90 | 1,676.87 | 100.92 | 77.4% |
| 8,192 | 512 | 5 | 1,419.99 | 77.41 | 1,424.85 | 77.39 | 60.7% |
| 32,768 | 512 | 5 | 1,156.10 | 70.11 | 1,157.94 | 70.06 | 62.5% |
| 65,536 | 512 | 5 | 940.65 | 57.07 | 941.85 | 57.00 | 60.4% |
| 131,072 | 512 | 5 | 665.12 | 41.85 | 665.77 | 41.77 | 55.1% |

Rates are tokens per second. All 25 measured requests used exact endpoint
token counts, completed the forced output length, and recorded zero prefix
cache hits. Full methodology and interpretation are in
[`../../RESULTS.md`](../../RESULTS.md). Each subdirectory contains the complete
JSON record for its five measured requests and discarded warm-ups.

Prompt-matrix SHA-256:
`01bea4cd9b0e1731629bb4df9927ae7cfafc84443c72521e3c849c96e95e884f`.

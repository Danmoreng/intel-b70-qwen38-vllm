# 2026-09-12 MTP6 production sweep

This directory contains the published result records for the final single-user
W4A16/MTP6 production profile. The large generated prompt matrix and raw SSE
captures are intentionally excluded from Git; they contain no private corpus,
but are unnecessary to reproduce the method and add about 10 MB.

## Results

| Input | Output | n | Client prefill median | Client decode median | Native prefill median | Native decode median | MTP acceptance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 128 | 5 | 1,763.01 | 117.85 | 1,788.27 | 117.80 | 65.6% |
| 8,192 | 512 | 5 | 1,924.94 | 88.66 | 1,932.17 | 88.63 | 45.5% |
| 32,768 | 512 | 5 | 1,535.58 | 70.81 | 1,538.91 | 70.75 | 45.6% |
| 65,536 | 512 | 5 | 1,209.07 | 69.49 | 1,211.08 | 69.39 | 53.1% |
| 131,072 | 512 | 5 | 785.40 | 50.54 | 786.32 | 50.43 | 48.7% |

Rates are tokens per second. Full methodology and interpretation are in
[`../../RESULTS.md`](../../RESULTS.md). Each subdirectory contains the complete
JSON record for its five measured requests and discarded warm-ups.

Prompt-matrix SHA-256:
`503093a13166f97960f377bf1a31c67583a70224478b47095d7e4d02dff88477`.

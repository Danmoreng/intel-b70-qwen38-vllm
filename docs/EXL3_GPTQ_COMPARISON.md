# EXL3 v1 and historical GPTQ comparison

These measurements belong to the qualified releases identified in the
[release report](EXL3_RELEASE_REPORT.md). Review candidates are not substituted
for measured release images.


GPTQ numbers are the published 2026-09-30 full run; EXL3 numbers are the new
complete run using identical frozen request payloads. The GPTQ full matrix was
not repeated. Its corrected v7 coding task above is a new paired measurement.
Both profiles run at 180 W; MTP depth and quantized checkpoints differ. Remaining
decode differences are shown explicitly alongside the quality/context gain.

| Actual input tokens | GPTQ prefill | EXL3 prefill | GPTQ decode | EXL3 decode | Decode change |
|---:|---:|---:|---:|---:|---:|
| 479–507 | 1694.3 | 1700.9 | 68.1 | 58.2 | -14.6% |
| 1,992–2,047 | 2271.4 | 2422.1 | 73.1 | 59.7 | -18.3% |
| 4,052–4,094 | 2118.4 | 2235.7 | 67.4 | 55.2 | -18.2% |
| 8,167–8,186 | 2004.6 | 2300.1 | 63.0 | 54.8 | -13.1% |
| 16,335–16,379 | 1816.3 | 2166.2 | 68.5 | 56.1 | -18.0% |
| 32,704–32,762 | 1783.6 | 1974.8 | 62.2 | 56.5 | -9.2% |
| 65,491–65,532 | 1560.9 | 1668.2 | 52.5 | 52.7 | +0.3% |
| 131,034–131,070 | 1224.0 | 1261.8 | 44.8 | 39.9 | -10.9% |
| 199,673–199,673 | 976.4 | 999.0 | 37.2 | 38.5 | +3.5% |

The parallel comparison uses aggregate output only while all requests overlap.
It preserves the original task mix and payloads; generated histories and
speculative acceptance can differ between checkpoints.

| Input budget / concurrency | GPTQ aggregate decode | EXL3 aggregate decode | Change |
|---|---:|---:|---:|
| 2,048 / C2 | 117.3 | 108.5 | -7.5% |
| 2,048 / C3 | 165.5 | 152.2 | -8.1% |
| 2,048 / C4 | 203.7 | 185.6 | -8.9% |
| 4,096 / C2 | 117.4 | 110.9 | -5.6% |
| 4,096 / C3 | 149.5 | 141.4 | -5.4% |
| 4,096 / C4 | 196.6 | 181.6 | -7.7% |
| 16,384 / C2 | 101.7 | 101.5 | -0.2% |
| 16,384 / C3 | 130.6 | 137.1 | +5.0% |
| 16,384 / C4 | 162.8 | 168.2 | +3.3% |


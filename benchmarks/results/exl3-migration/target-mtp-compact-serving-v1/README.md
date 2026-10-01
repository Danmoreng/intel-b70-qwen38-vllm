# Compact MTP3/MTP4 serving screen

One historical MTP3 pass and one fresh MTP4 pass, 36 waves / 84 requests. This is screening, not a completed ABBA or final release qualification.

C1 uses native request-weighted decode; C4 uses client aggregate decode over the union of observed request decode intervals. Both are serving observations, including scheduling/prefill interference. Actual cache hits are shown because a warm label does not establish identical residency.

| Context / workload | C | Cache | MTP3 decode tok/s | MTP4 decode tok/s | Change | Cache hits 3 / 4 |
|---|---:|---|---:|---:|---:|---:|
| code-4096 | 1 | cold | 69.29 | 93.67 | +35.2% | 0.0% / 0.0% |
| code-4096 | 1 | warm | 69.38 | 80.03 | +15.3% | 39.1% / 40.6% |
| code-4096 | 4 | cold | 149.97 | 158.37 | +5.6% | 0.0% / 0.0% |
| code-4096 | 4 | warm | 157.75 | 166.57 | +5.6% | 39.1% / 40.6% |
| prose-4096 | 1 | cold | 52.15 | 57.76 | +10.7% | 0.0% / 0.0% |
| prose-4096 | 1 | warm | 57.88 | 48.50 | -16.2% | 39.1% / 40.6% |
| prose-4096 | 4 | cold | 121.91 | 112.29 | -7.9% | 0.0% / 0.0% |
| prose-4096 | 4 | warm | 125.73 | 116.13 | -7.6% | 39.1% / 40.6% |
| code-49152 | 1 | cold | 60.36 | 69.00 | +14.3% | 0.0% / 0.0% |
| code-49152 | 1 | warm | 60.37 | 65.80 | +9.0% | 94.4% / 94.8% |
| code-49152 | 4 | cold | 15.35 | 15.27 | -0.5% | 0.0% / 0.0% |
| code-49152 | 4 | warm | 86.27 | 20.64 | -76.1% | 94.4% / 47.4% |
| prose-49152 | 1 | cold | 45.32 | 47.38 | +4.6% | 0.0% / 0.0% |
| prose-49152 | 1 | warm | 45.32 | 41.20 | -9.1% | 94.4% / 94.8% |
| prose-49152 | 4 | cold | 14.77 | 14.58 | -1.3% | 0.0% / 0.0% |
| prose-49152 | 4 | warm | 70.86 | 19.51 | -72.5% | 94.4% / 47.4% |
| code-102752 | 1 | cold | 43.94 | 44.40 | +1.1% | 0.0% / 0.0% |
| code-102752 | 1 | warm | 43.90 | 44.38 | +1.1% | 98.1% / 97.2% |

Decision after component profiling and generated-path diagnostics: use MTP3 as the EXL3 target-profile default for subsequent optimization. MTP4 gains only~1.1% at103K C1 and regresses about8% on4K prose C4; warm48K C4 also has unequal cache residency. No further depth matrix is planned. This practical default selection does not complete the production release gates.

Exact output hashes and first differing token positions are retained in assessment.json. Sequence variation is not assumed to be harmless numerical rounding. Instrumented component traces remain separate from these throughput observations.

All 84 requests completed with zero observed preemption. The fresh compact
controller ran for 978.15 seconds, including MTP4 startup and excluded warmup;
the 18 reused MTP3 waves are historical and are not part of that elapsed time.
Only 3 of 42 paired request outputs are token-identical between depths.
Some prose outputs differ at the second generated token. These variations
require an actual generated-path/margin check before release; the short and long
teacher-forced quality panels do not establish this property.

At warm 48K C4, MTP3 reused 185,600 prompt tokens, MTP4 only 93,184, from
196,608 submitted prompt tokens per wave. The large serving-rate difference
includes additional prefill work and decode/prefill interference. Both startup
logs report 10.44 GiB available KV memory; automatic page sizes differ
(1600/1664), as do reported token capacities (294,362/287,464). This alone does
not establish the exact eviction cause; retain it for targeted cache diagnosis.

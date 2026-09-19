# E03 decision: stop after region-cost measurement

Measured the unchanged production M04 binary, not a new kernel. The review makes implementation conditional on profile evidence. Removing the output-layout work entirely in a deliberately incorrect diagnostic did not show a substantial gain even within the M04 region. No C++ or production adapter change is justified by this screen.

| M | Context | Full region, graph (ms) | Estimated removable time (ms) | Region reduction |
|---:|---:|---:|---:|---:|
| 2 | 8192 | 0.1266 | 0.0016 | +1.23% |
| 2 | 16384 | 0.1724 | 0.0024 | +1.42% |
| 2 | 65536 | 0.4435 | 0.0017 | +0.39% |
| 2 | 196608 | 1.1679 | 0.0020 | +0.17% |
| 3 | 8192 | 0.1040 | -0.0003 | -0.30% |
| 3 | 16384 | 0.1554 | 0.0025 | +1.64% |
| 3 | 65536 | 0.5114 | 0.0012 | +0.24% |
| 3 | 196608 | 1.4798 | 0.0052 | +0.35% |
| 4 | 8192 | 0.1210 | -0.0001 | -0.12% |
| 4 | 16384 | 0.1740 | 0.0020 | +1.14% |
| 4 | 65536 | 0.5342 | 0.0018 | +0.34% |
| 4 | 196608 | 1.5087 | 0.0048 | +0.32% |
| 5 | 8192 | 0.1497 | 0.0016 | +1.04% |
| 5 | 16384 | 0.2242 | 0.0026 | +1.16% |
| 5 | 65536 | 0.7477 | 0.0077 | +1.03% |
| 5 | 196608 | 2.1161 | 0.0107 | +0.51% |

Both arms use the same Q/K/V, random physical page mapping, FP8 production strides, fixed `{2:32, 3:8, 4:16, 5:16}` split policy and `max_seqlen_k=200704`, matching the observed FULL-graph bound. Actual `seqused_k` varies by context. 21 alternating randomized blocks of 10 calls per arm in eager and graph modes. The full region includes input packing, native attention/reduction, output permutation and final copy. Scratch buffers are preallocated in both arms; eager host allocation cost is not being estimated.

The separate layout-only timing is not added to the full region: isolated launch costs are not generally additive. The empirical difference between full and elided graphs is an optimistic diagnostic, not a rigorous mathematical upper bound or a valid implementation. A correct fused store could itself have overhead. There is no measured E03 serving gain, and no long benchmark was run.

The original output mapping was checked for M=2,3,4,5; no new candidate correctness claim is made. Native binary SHA-256: `784916abd42b614794becac634a6a154f854c73d6dfcb1a50cd939b563020c6a`. Image: production `b675d81d4e7cc63fbcd6df395965ea16ec5c4704428c81118a1618185245dd5a`. Raw evidence: `runs/layout-cost-20260919-203934`. Compact data: `results.json`.

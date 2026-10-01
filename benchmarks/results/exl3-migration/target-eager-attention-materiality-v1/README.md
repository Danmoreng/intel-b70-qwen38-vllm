# Eager attention materiality probe

One103K code C1/MTP3 request generated128 tokens; only eight pure-decode
cycles were traced. Same immutable image and checkpoint, FP8 KV, INT8 prefill,
quota.965 and262K contract; graph capture was explicitly disabled for this
diagnostic. The original serving path remains unchanged.

Kernel correlations are joined to driver/runtime external IDs, CPU operator
records and the enclosing per-cycle stage ranges. This attributes128 target
attention main kernels plus128 reductions, and24 draft main kernels plus24
reductions, to the eight traced cycles. Those counts agree with16 target
attention layers and three single-layer draft steps per cycle.

| Eager stage | Summed GPU kernel ms / cycle | Attention main + reduction ms / cycle | Attention share of stage kernel sum |
|---|---:|---:|---:|
| Target body | 54.22 | 21.99 | 40.56% |
| Complete draft proposal | 6.20 | 2.86 | 46.12% |

These are summed kernel durations, **not wall-clock timings or a measured
serving speedup**. Graph/host overhead and any overlapping work must not be
inferred from these percentages. The evidence supports a guarded M04 prototype
after numerical gates; actual serving-graph and C4 benefits must still be tested.
Raw57MB tracing output stays outside tracked/README fixtures; its SHA-256 and
lean attributed totals are retained in assessment.json.

The eager output hash differs from the serving graph output's first128 tokens.
The two graph-based component depth arms matched on this128-token code prefix,
but that does not establish equality with eager execution. The eager diagnostic
stored an output hash rather than token IDs, so an exact divergence position
cannot be reconstructed from it. Generated-path margin/logprob checks are the
next correctness step; do not label the variation harmless without evidence.
No M04 route, kernel or production setting has been changed.

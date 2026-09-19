# E04 decision: no weight-cache implementation on present evidence

The review requires a measured, relevant remaining weight transformation before
implementing a cache. The reference serving trace and actual TorchInductor cache
show `weight.float()+1` embedded in the existing norm kernels. Representative
generated kernel definitions and source hashes are in `compiled-evidence.json`.

This proves that the Python expression does not imply a separate launch. It
does not prove that the inline operation has zero arithmetic cost. The trace
does not expose decode graph interiors either. There is no measured bottleneck
here that justifies new FP32 buffers and invalidation/weight-loading machinery.

No weight dtype or original parameter was changed. No candidate was built,
no serving performance claim is made, and nothing was promoted. Reopen only
with evidence that the remaining transformation is materially expensive.
See [profile observations](../PROFILING.md).

# oneDNN SDPA host constant cache — isolated gates passed

EXL3 commit d181d97 removes the per-call device scalar read and caches supplied host scale values as persistent device constants, scoped to worker/thread and actual SYCL queue. Compiled oneDNN partitions share that same owner. Invalid, nonfinite or non-FP16-representable inverse scales fail before execution, using integer bit checks that remain valid under the existing fast-math compiler flags.

Both old/new artifacts pass49 dense FP32 comparisons, including padded valid-query invariance and a65537-key case. Maximum absolute error is0.000880957 in both arms. The new artifact additionally rejects seven invalid scales and passes two independent worker/thread/stream sequences with alternating scales. On36 warmed calls, CPU profiler device-scalar reads fall36→0. The four-head256Q/4096K call envelope falls from1.44→0.97ms median wall time (GPU event window1.11→0.65ms); these windows include host launch/synchronization gaps and are not pure GPU compute time or end-to-end engine gains.

Fresh image: sha256:af332911245f71b020e7bf2abb5a28f2849d98d98b8df0c1fa1ed477a772a0f0.
Native library:48f879c16f3695dbb0fae7388f1e318591d6408acc7be15b0093a0bbb6d9053b.
Build/ABI/source/import receipts and raw SHA hashes are retained. No serving default or production change. The unified exact-length prefill/mixed dispatcher and final capacity/quality/tail qualification remain open.

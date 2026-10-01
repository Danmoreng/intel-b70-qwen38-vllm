# EXL3 Shared-KV verification prototype

The separate native extension compiles and CPU-loads against Torch2.13.0+xpu/SYCL9 on the pinned oneAPI2026.1.1 builder. It does not reuse the old GPTQ binary and does not modify the serving image. The original pinned kernel headers are patched with the reviewed packed-row causal rule; actual EXL3 page size and KV strides are passed at runtime. Queries from a uniform multi-sequence batch are packed together for one attention kernel invocation, rather than a Python loop of C1 launches. KV update remains the native runner's responsibility; this operation only reads KV.

CPU guard/packing checks pass51 supported/fallback cases. The prepared XPU probe covers q2/3/4/5, pages64/1600/1664, exact boundaries, non-unit FP8 scales, randomized/disjoint C1/C4 pages, an independent dense FP32 oracle, future-token poisoning, unchanged KV bytes, and graph replay after device lengths/page tables/query values change. Unsupported q1/6/63/64/255/256 delegates to native. Tolerance is the historical M04 rtol0.01/atol0.003, fixed before collection. Long timing includes pack/unpack and shared batched launches; it remains a microbenchmark.

Build/import is proven; XPU correctness and actual graph serving benefit are unproven. No deployment/default switch is authorized by these tests. GPTQ stays offline. Raw binaries/traces remain outside tracked fixtures.

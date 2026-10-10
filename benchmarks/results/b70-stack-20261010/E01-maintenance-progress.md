# E01 stable oneDNN maintenance — qualification in progress

Original loaded custom library ee8fc42... maps to clean local oneDNN source
0e2a5bfe..., original GPU SYCL/CPU NONE/Graph ON build. Its embedded micro.cl
retains initialization after the early barrier-arrive. Stable v3.13.4 source
c531fe1... has the corrected initialization before barrier-arrive. Audit and
source excerpts are retained in oneDNN-provenance-r2; this is no longer an
unresolved vendor/dirty-build assumption.

Stable library SHA256 `90c0feddfb51a6219903d5e1ccabc9fd32dab0c702284a16dd7692bb6042e056`; native EXL3 rebuild SHA256
`00b65587220481ec89269f7510136146809d7e01938de3d30d5b67206948a95a`.
Torch 2.14.0+xpu, SYCL9/oneAPI26.1.1, native flags and EXL3 source commit
fc8f8c9... retained. Candidate image b772d379... is an experimental child of
the exact qualified image, preserving installed MTP policy and source guards.

Artifact registration/import, 31 attention numerical cases, 3 causal poison
cases, 2 mutated graph suites and real-weight W8A8 prefill reference PASS.
Ordinary creation logging confirms GPU SDPA `ocl:micro:reusable` and fused
`sdp_primitive_kernel_t` on FP16 Q=256/K=4097/GQA6/D256, strict fpmath.
No execution verbose profiler was requested from the queue.

81-key, three-epoch application cache soak: capacity64, peak64, 177 evictions,
488 hits. Post-warmup growth: RSS 749568
bytes, GPU allocated/reserved both zero growth. This finite soak is not a
universal proof about all driver/oneDNN allocations or in-flight pressure;
pressure_waits was zero. Near-limit serving and paired throughput remain
pending. No release promotion or decode-speedup claim yet.

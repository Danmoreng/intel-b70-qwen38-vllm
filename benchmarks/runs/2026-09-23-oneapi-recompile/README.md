# oneAPI 2026.0 versus 2026.1.1 for Q128/M04

On 2026-09-23, Q128 and M04 were rebuilt against the same vLLM 0.30.0
production image (`sha256:cd6562f03c8328fe60ca69269d0e4175a284859e56525950fbfc021be19d73f3`),
the same pinned `vllm-xpu-kernels` and SYCL-TLA source commits, and the same
compiler flags. Q128 was built from unpatched `docker/q128/tiles.cpp`; the
unchanged shared-KV patch was applied only before compiling M04. The only
intended build-arm difference was Intel oneAPI DPC++/C++ Compiler 2026.0.0
versus 2026.1.1. Both pairs linked to the same torch and SYCL library SONAMEs.
The compiled artifacts are kept locally under
`benchmarks/experiments/m17-oneapi-recompile/runs/build-20260923-170449`;
they were not installed in the production service.

The installed 2026.1.1 compiler came from Intel's signed APT repository in a
separate builder image. Its package version was `2026.1.1-325`; the builder
image ID was `sha256:ae6950731b3c031f812a95c0eb615a572239426440bf67fd1b3c9c9cbfce9eca`.
The production image and service remained unchanged.

## Operator gate

The same seeded FP16 queries and randomly quantized FP8 K/V with **production
interleaved stride** `(3407872, 2048, 512, 1)` were used for both compilers.
Q128 used 6,656 query tokens and M04 used a four-token verification batch with
the production 16-split policy. Both were checked against native attention at
64K and 192K. Q128 matched exactly; M04 passed `allclose` with maximum absolute
error below `8e-6`. One process per binary pair avoided duplicate Torch op
registrations; runs alternated 2026.0 / 2026.1.1 / 2026.1.1 / 2026.0.
Each run had three warmups and eight device-event samples per shape.

| Operator | KV | 2026.0 median | 2026.1.1 median | New compiler speedup |
|---|---:|---:|---:|---:|
| Q128 | 64K | 294.09 ms | 294.23 ms | -0.05% |
| Q128 | 192K | 931.10 ms | 929.16 ms | +0.21% |
| M04 | 64K | 0.5950 ms | 0.5984 ms | -0.57% |
| M04 | 192K | 1.6702 ms | 1.6623 ms | +0.47% |

Medians pool the 16 device-event samples per arm and shape across the two
passes. Individual run medians and samples are in [`v0-r1.json`](v0-r1.json),
[`v1-r1.json`](v1-r1.json), [`v1-r2.json`](v1-r2.json) and
[`v0-r2.json`](v0-r2.json). The separately measured deployed binaries were
also within the small run-to-run range: see [`deployed.json`](deployed.json).

## Decision

No practical performance improvement was found. The effects are below 1%,
change sign across the two contexts, and do not meet the existing 2% operator
gate. No candidate serving image or full benchmark was started. Keep the
deployed binaries and oneAPI 2026.0 build provenance. A future compiler-only
retest should use a new release or a specific code-generation hypothesis.

Build scripts and the exact operator workload are under
[`m17-oneapi-recompile`](../../experiments/m17-oneapi-recompile/).

# PTI native-graph lifetime fix

The tested Torch 2.15 / PTI 1.2.1 stack replays incorrect graph outputs after
XPU profiling. A small standalone PyTorch reproduction needs no vLLM,
EXL3, Triton compilation or model weights.

Intel's [upstream fix](https://github.com/intel/pti-gpu/commit/dd8940ff817f94461795708fe3c386ab9e09d3b1)
cleans up profiler-owned native graphs and stale handles when graphs are
destroyed or tracing is disabled. The source revision embedded in the
published 1.2.1 library predates this fix. The direct build applies its four
runtime-file changes to that exact source; tests and CI changes are omitted.

Torch remains the unmodified official wheel. Native graph recording and
per-kernel replay profiling stay enabled. The unchanged original matrix,
attention and GDN gates pass with the backport; the standalone replay test
also records all nine expected GPU kernel events. Final-image quality and
serving qualification is tracked in the
[test evidence](../../../benchmarks/results/direct-xpu-20261114/README.md).

[upstream.json](upstream.json) pins source files before/after the patch,
the upstream commit, patch hash and build dependencies. [build.py](build.py)
checks these identities, builds with two jobs, and records library hashes.
Only PTI's interface and core library enter the runtime image; the selected
Level Zero loader and Intel userspace are retained. PTI's MIT license is
copied into the image. No precompiled local binary is a build input.

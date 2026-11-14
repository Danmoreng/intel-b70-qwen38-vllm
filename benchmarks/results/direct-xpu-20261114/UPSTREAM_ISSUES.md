# Related upstream reports

Checked 2026-11-14 in public PyTorch, Kineto, Intel PTI and torch-xpu-ops.

The matching upstream correction is Intel PTI's
[Fix Native Graph Tests](https://github.com/intel/pti-gpu/commit/dd8940ff817f94461795708fe3c386ab9e09d3b1).
It cleans up native graphs and stale handles when graphs are destroyed or
tracing stops. The installed PTI 1.2.1 source revision lacks it. A matched
rebuild without the change reproduces the bad two-row result; the four-file
backport passes the unchanged full matrix and records all nine kernels in
native graph replay profiling. See [controlled evidence](pti-fix-controls.json).
No new issue or comment was posted upstream.

The following reports are related but are not this confirmed correction.

| Report | Observed scope | Why it is not an exact match |
|---|---|---|
| [PyTorch #189403](https://github.com/pytorch/pytorch/issues/189403) | XPU graph replay with captured all-reduce can corrupt later eager all-reduce; standalone reproduction outside vLLM. Torch 2.11, multi-GPU/XCCL. | Our failure is a finite wrong matrix result on one GPU, without distributed collectives. |
| [PyTorch #197520](https://github.com/pytorch/pytorch/issues/197520) | `empty_cache()` after a heavy kernel chain produces NaNs on Arc A730M/Windows; an Intel contributor also reports DEVICE_LOST on A770M with 2.14 and a 2.15 nightly. | Different trigger, OS, hardware and failure signature. This does not establish our bug on Alchemist. |
| [PyTorch #203535](https://github.com/pytorch/pytorch/issues/203535) | Non-tree CUDA graphs with donated buffers yield incorrect first-backward gradients on NVIDIA L4, including Torch 2.15. | Training/backward and donated-buffer reuse, rather than our inference/native-matrix replay. |
| [PyTorch #171689](https://github.com/pytorch/pytorch/issues/171689) | Missing profiler events during XPU graph replay. Maintainers say this should work in 2.15. | Missing trace records are not evidence of incorrect arithmetic. |

The official [2.15 XPU feature report](https://github.com/pytorch/pytorch/issues/200933)
announces kernel-level profiling visibility into XPU graph replay. It is a
relevant area to investigate because our failing sequence includes profiling,
but timing and symptom similarity do not identify the defective upstream component.

Subsequent controlled runs retained all five extra compiled invocations,
synchronization, garbage collection and the original numerical thresholds:

- Removing only the profiler and trace export passed all seven shape classes.
- Keeping the profiler with CPU activities only also passed all seven classes.
- XPU-only profiling still reproduced the same gate-up two-row failure.
- Omitting the between-group `empty_cache()` did not prevent the failure.
- A trial synchronization after `_disable_profiler()` did not fix the unchanged gate.

These initial controls establish XPU profiling as a required part of the
sequence; the later matched PTI rebuild/backport comparison isolates the
effective correction. Diagnostic edits to tests are not release qualification.

The regression is independently reproducible with simple PyTorch operations,
without vLLM, EXL3, Triton compilation or checkpoint data. Source-level lifetime
code is shared, but actual reproduction on another GPU remains unverified.

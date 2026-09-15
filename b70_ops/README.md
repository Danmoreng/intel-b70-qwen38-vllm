# b70_ops

Reproducible, narrow B70 operator experiments. Version 0.1 exposes the deployed
Q128/KV32 prefill specialization under a stable `b70_ops` namespace with both
an allocating control API and a caller-output API.

- `q128_forward` allocates its output.
- `q128_forward_out` writes directly into the tensor supplied by vLLM.

The second API removes the adapter's `out.copy_(temporary)` without changing
the Q128 kernel or its dispatch guard. This is not a production claim until
real-activation correctness and alternating serving measurements pass.

Run `./b70_ops/build.sh`. The script verifies the exact compiler, runtime image,
vLLM-XPU-kernels and SYCL-TLA inputs pinned in `sources.lock.json`, then stores
the library, build log and checksum in `b70_ops/build/`.

# Production M04 shared-KV verification

The default image combines this adapter with `../q128/tiles.so` for prefill
and `m04.so` for MTP verification. Both libraries and the adapter match the
running Q128 + M04 production image used for the published coding benchmark.

| Artifact | SHA-256 |
|---|---|
| `m04.so` | `784916abd42b614794becac634a6a154f854c73d6dfcb1a50cd939b563020c6a` |
| `b70_attention.py` | `d288bb54f52f5be166e851b2d7862da81c041bd7eff6b90e6eaca68c92333b64` |

M04 uses packed query tiles of eight rows for eligible 2–5-token verification
calls. Q8 describes the query tile size, not weight or activation precision.
The adapter dispatches eligible larger prefills to Q128/KV32; unsupported
signatures use native attention. The model's target output head stays FP16.

The library is prebuilt for the exact base-image digest in
[`../Dockerfile`](../Dockerfile), vLLM 0.29.0+xpu and XPU kernels 0.1.14.1.
The build verifies the library and adapter hashes. Do not change the ABI
without rebuilding and validating the extension.

Local operator sources live in [`../../b70_ops/csrc`](../../b70_ops/csrc),
with the shared-KV upstream-header changes in
[`shared-kv-verification.patch`](../../b70_ops/patches/shared-kv-verification.patch).
Compiler flags are in [`build-container.sh`](../../b70_ops/build-container.sh);
upstream source commits are in [`sources.lock.json`](../../b70_ops/sources.lock.json):
oneAPI 2026.0.0, vLLM-XPU-kernels `6d92b1bfbf32767ecda8e819613eb151e70030ad`,
SYCL-TLA `87f6850680a580654b9ea2c80dbc01aeb36ad231`.
The operator development build wrapper requires separately provisioned compiler,
source trees and its pinned local build image. Ordinary deployment uses the
included, hash-checked production binary and needs none of those host paths.

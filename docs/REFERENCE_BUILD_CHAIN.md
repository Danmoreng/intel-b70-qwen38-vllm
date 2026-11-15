# Previous reference image: historical build chain

[Back to installation](INSTALLATION.md)

This records how the preceding v7 reference image was assembled. The current
v8 production recipe uses the [direct XPU build](../engine/direct-xpu/README.md).

The build is a staged reconstruction requiring compiled dependencies and local
parent images. The following map makes the prerequisites explicit; it is not a
single-command clean-machine installer.

| Stage | Inputs and instructions | Result |
|---|---|---|
| 1. Compiler/base | [Base Dockerfile](../docker/Dockerfile), pinned upstream images; `B70_BASE_BUILD_IMAGE=local/b70-qwen38-vllm:base-development ./scripts/build-image.sh` | vLLM 0.31 / Torch 2.14 compiler/base image |
| 2. Native EXL3 and M04 | [Native bootstrap](../engine/exl3xpu/README.md), extracted source archive, compatible builder/base images, verified Torch-2.14 M04 artifacts and oneDNN 3.13.4 installation | Native 0.31 EXL3 candidate |
| 3. vLLM 0.33 adapters | [Python overlay workflow](../engine/exl3xpu/README.md), the preceding native base and SHA-verified official XPU wheel | 0.33 candidate retaining the native ABI |
| 4. MB4/K4 kernels | [Matrix build and image preparation](../engine/exl3xpu/mb4-k4/README.md), pinned native builder, oneDNN headers/library, same-binary numerical and matrix-panel receipts | Guarded matrix candidate and source-attested release layer |
| 5. Fine-prefix cache | [Cache layer](../engine/cache1792/README.md), the pinned v6 parent image | 1792/128 cache layer with the automatic COW correction |
| 6. Validate and package | [Historical v7 qualification](../benchmarks/results/mtp-cache-1792-20261112/README.md), current configuration and release-specific serving, numerical, capacity and operational gates | Independently qualified manifest, image and runtime overlay |

The matrix image preparation and cache Dockerfiles pin local parent IDs.
Stage 5's command only works when that exact parent is present. An independent
rebuild must reconstruct and review its parent chain, update the corresponding
candidate inputs and record new identities; re-tagging a different image as
production cannot satisfy the launcher's hash checks. The native bootstrap also
requires M04 and oneDNN build artifacts that the repository does not ship as
binaries. Existing build/gate receipts are evidence for the reference build,
not substitutes for missing inputs or fresh validation.

Keep development tags separate from production and rollback tags. The base
builder rejects protected release aliases. The [saved v7 manifest](../config/releases/exl3-v7/production_image.json)
identifies this historical reference. It is the immediate rollback for v8.

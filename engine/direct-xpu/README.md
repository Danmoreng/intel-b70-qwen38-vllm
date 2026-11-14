# Direct vLLM XPU build

Build directly on the **official vLLM 0.33.0 XPU image**. No 0.31 image,
local parent image or previously compiled native library is required. The
build compiles oneDNN, EXL3 MB4/K4 and M04 from public sources and installs the
current Python integration. Model weights are mounted separately for testing.

**Status: candidate, not a qualified replacement.** The deployed engine and
its release manifests remain unchanged. Build/import checks must be followed
by numerical GPU, model, quality, cache, capacity and serving measurements.

## Build

Requirements: Linux x86-64, Docker, Python 3.12+, network access to the public
sources, Python indexes and Intel APT repository, and space for the upstream
image, compiler stage and build outputs. The wrapper caps containers at 6 GiB
and four CPU cores; oneDNN compilation uses four jobs. It mounts no GPU devices
or model weights during the build.

From the repository root, choose a fresh tag and output directory:

```bash
python3 scripts/build-engine.py \
  --tag local/b70-qwen38-vllm:direct-xpu-development \
  --output benchmark-results/direct-xpu-development
```

`--prepare-only` downloads and verifies inputs without Docker. Optional
`--download-cache PATH` reuses existing wheels after checksum verification.
The Docker-build deadline defaults to three hours (`--timeout-seconds` can
reduce it). The wrapper refuses protected/existing tags and builds whose
deadline plus cleanup margin would enter the reference host's 22:45
Europe/Berlin closing phase. Failures retain logs and remove their own build
containers. The wrapper does not change the service or publish an image.

## Selected components

The policy is to test the newest stable components first. Retain a newer
reference component if an upstream image ships an older one; select an older
version only after recording a regression. Pins below identify a reproducible
candidate selection, not a policy of retaining old dependencies.

<!-- BEGIN DIRECT STACK -->
| Component | Candidate |
| --- | --- |
| vLLM / base image | 0.33.0 XPU, pinned registry digest |
| PyTorch / Triton XPU | 2.15.0+xpu / 3.9.0 |
| vLLM XPU kernels | 0.1.17.1 |
| Intel compiler package | `intel-oneapi-compiler-dpcpp-cpp-2026.2=2026.2.0-863` |
| SYCL runtime / PTI | 2026.2.0 / 1.2.1 |
| Intel Compute Runtime / IGC / GMM | 26.40.40196.8 / 2.42.6 / 22.10.2 |
| Custom oneDNN source / reported version | 3.14.2 / 3.14.1 |
| Transformers / xgrammar | 5.22.2 / 0.2.9 |
<!-- END DIRECT STACK -->

Torch 2.15 uses a [source-pinned Intel PTI fix](pti/README.md) for the
demonstrated native graph/profiling regression. The official Torch wheel,
native graph recording and kernel-level replay profiling are retained.
The selected profile also disables **Inductor model compilation** (`mode: 0`),
while retaining `FULL_DECODE_ONLY` native graphs. The default compiled
Torch 2.15 path reproducibly degenerates in the frozen coding request;
matched no-compilation controls do not. The exact faulty optimization is
unproven. This is a compatibility fallback with full quality/performance
qualification pending, not a claim that the PTI fix repairs every compiler bug.
Full qualification is tracked in the
[test evidence](../../benchmarks/results/direct-xpu-20261114/README.md).
Torch is installed without dependency downgrades:
the reference's newer CCL is retained, and runtime/PTI upgrades are explicit.
Transformers 5.23 fails the existing video API contract; the otherwise identical
5.22.2 control passes. The build selects 5.22.2, still newer than the reference
5.17, and records this measured compatibility exception in the lock.
These choices exceed package metadata constraints; `pip check` is not clean.
GPU tests, rather than successful imports, determine compatibility.
The host's kernel/DRM driver is unchanged; Intel userspace components are
installed inside the image. The newer GMM also satisfies the inherited
VA-API package, so it remains installed.

The official oneDNN v3.14.2 source tag still declares internal version 3.14.1
in its CMake file. The source is preserved unchanged; the lock and CPU checks
record both identities separately.

[build-lock.json](build-lock.json) records versions, archive commits, download
checksums, compiler selection and pristine vLLM hashes. The latest stable
vLLM check on 2026-11-14 found 0.33.0: its official
[release](https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3) has an
rc3 tag but is marked `prerelease=false` and supplies final 0.33.0 artifacts.
The [0.1.17 kernel branch](https://github.com/vllm-project/vllm-xpu-kernels/releases/tag/v0.1.17)
targets Torch 2.15. The Intel userspace is from
[Compute Runtime 26.40](https://github.com/intel/compute-runtime/releases/tag/26.40.40196.8).

## Source layout

- [source-manifest.json](source-manifest.json): readable integration exported
  from the qualified reference, file hashes, and explicit rebases for the
  newer Torch guard and M04 header API. No native binaries or weights.
- [Dockerfile](../../docker/direct-xpu/Dockerfile): upstream, runtime, native
  builder and final candidate stages; no vLLM upgrade chain.
- [build-native.sh](build-native.sh): oneDNN, existing
  [matrix inputs](../exl3xpu/mb4-k4/source), and [M04 source](m04/csrc/shared_kv_verification.cpp).
  M04 retains the packed causal mask and Python operator signature, and uses
  the current XE2 namespace and unified FP32 LSE scratch. Header/SYCL-TLA
  inputs match the selected XPU kernel release.
- [install.py](install.py): rejects unknown upstream runner/MTP boundaries,
  installs the current adapters and automatic complete COW helper, and binds
  native guards and profile hashes to the newly compiled artifacts.
- [check-image.py](check-image.py): CPU checks for versions, ABI, guarded
  source boundaries, native registration and integration contracts.

APT repositories and transitive Python resolution are not snapshots. The
build records the installed package lists instead of claiming bit-identical
rebuilds. Source and binary integrity checks do not qualify GPU correctness.

## Qualification

Build output records the context inventory, archives, command, logs and image
ID. The image retains compiler/native manifests under `/native/`, source locks
and `/opt/b70-direct/installation.json`.

Then use exclusive GPU access for the checks described in
[current qualification](../../docs/BENCHMARKS.md): original numerical
tolerances, matrix/M04 operators, MTP3, complete COW copies, 1792/128 prefix
reuse, genuine RAM restore, long-context admission, media/API/tools and
abort/restart. Compare the frozen serving matrix and natural tool-agent tasks
against the exact deployed image before promotion and publication.

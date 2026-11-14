# Installation and source builds

[Back to README](../README.md)

## What is available

This is a reference-host recipe with strict release checks. The repository
publishes source, configuration and evidence, but does not distribute the
qualified Docker image, native binaries, compiler caches or large benchmark
fixtures. A `local/...` image name is a local tag, not a public registry image.
An image ID beginning with `sha256:` identifies a local Docker image; it is not
by itself a pullable registry address.

There are two paths:

- **Existing qualified installation:** use the exact image identified by
  [the current manifest](../config/production_image.json) and the setup below.
- **New source build:** follow the stages below with the required native build
  inputs. A new image needs its own validation and release manifest. Cloning
  this repository and running `build-image.sh` alone does not produce the
  qualified current engine.

## Host requirements

- A Linux x86-64 host with an Intel Arc Pro B70, 32 GB VRAM, working `xe`/DRM
  device access and Docker. The service user needs access to Docker and the GPU.
- Python 3, Bash, Git, systemd user services, `curl`, `jq` and GNU coreutils.
  On Ubuntu/Debian, install the small CLI dependencies with
  `sudo apt install python3 git curl jq coreutils` if needed.
- Host RAM headroom: the launcher requires at least the chosen RAM-cache budget
  plus 7 GiB available before loading. The default is 12 GiB of cache; the
  container has a fixed 23 GiB host-memory limit and no swap allowance.
- Disk space for the checkpoint, Docker images, native build inputs and
  policy/image-specific compiler caches. These artifacts are not in Git.
- Permission to set the GPU power cap. The power helper and strict launcher
  currently check PCI device `0000:03:00.0`; the default render node is
  `/dev/dri/renderD128`. Inspect the host topology before using these scripts
  elsewhere. `RENDER_NODE` can be set in `.env`; the PCI power path is currently
  in Python source and is not an `.env` option.

The pinned Intel userspace runtime and compiler live in the image. The exact
versions are in [current configuration](CURRENT_CONFIGURATION.md); they do not
replace the host kernel's GPU support. Portability to other host configurations
has not been qualified by the reference-host receipts.

## Set up an existing qualified image

Clone the public repository on a new installation, then work from its root:

```bash
git clone https://github.com/Danmoreng/intel-b70-qwen38-vllm.git
cd intel-b70-qwen38-vllm
cp .env.example .env
```

For an existing checkout, retain its `.env`. The service installer records the
absolute checkout path; recovery must run from that same checkout.

Confirm that the exact image is available locally before downloading weights:

```bash
python3 - <<'PY'
import json, subprocess
release = json.load(open('config/production_image.json'))
actual = subprocess.check_output(
    ['docker', 'image', 'inspect', release['image_tag'], '--format', '{{.Id}}'],
    text=True).strip()
assert actual == release['image_id'], 'Local image does not match the release'
print(actual)
PY
./scripts/download-model.sh
```

The downloader reads the model revision and immutable image ID from the
manifests. It writes the EXL3 checkpoint under
`~/.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw`; serving mounts it read-only.
`.env` model/image overrides do not redirect this pinned EXL3 download.

Set the 180 W cap with permission to write the host's `power1_cap`, then install:

```bash
./scripts/set-power-limit.py
python3 scripts/run-server-exl3.py --check-only
./scripts/install-user-service.sh
```

The power helper does not elevate privileges. If the power-control file is
root-owned, an administrator must set the cap or grant appropriate access.
The installer starts the engine and enables its watchdog. Model loading is
asynchronous; check `curl -fsS http://127.0.0.1:8081/health` until it succeeds,
then make the [example request](../README.md#make-a-request).

## Source-build stages

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
| 6. Validate and package | [Current qualification scope](BENCHMARKS.md), current configuration and release-specific serving, numerical, capacity and operational gates | Independently qualified manifest, image and runtime overlay |

The matrix image preparation and cache Dockerfiles pin local parent IDs.
Stage 5's command only works when that exact parent is present. An independent
rebuild must reconstruct and review its parent chain, update the corresponding
candidate inputs and record new identities; re-tagging a different image as
production cannot satisfy the launcher's hash checks. The native bootstrap also
requires M04 and oneDNN build artifacts that the repository does not ship as
binaries. Existing build/gate receipts are evidence for the reference build,
not substitutes for missing inputs or fresh validation.

Keep development tags separate from production and rollback tags. The base
builder rejects protected release aliases. Published source pins and the
[current manifest](../config/production_image.json) are the authority for the
reference build; no clean-host reconstruction was performed for this
documentation consolidation.

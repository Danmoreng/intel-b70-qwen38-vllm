# Native EXL3 source and vLLM 0.33 integration

The complete source archive preserves the upstream license. The manifest pins
upstream/base, source commit, all 74 build inputs,
native compiler/ABI identities and the previous qualified 0.31 image.
Weights, native binaries, caches and large numerical arrays are omitted.
Prior release configuration and source snapshots remain under `config/releases`.

Current production also uses the [MB4/K4 patch and compiled source inputs](mb4-k4/README.md).

This archive supplies the unchanged native bootstrap. Current production adds
the separately guarded [vLLM 0.33 Python port](../../benchmarks/experiments/vllm-033-upgrade/README.md).
The archive alone does not contain the current Python runtime. The
[current release manifest](../../config/production_image.json) records both
native artifact identities and the Python port; the original native source
manifest remains historical provenance.

Extract `source.tar.gz`, or apply `migration.patch` to the recorded upstream
base. The release child and runtime parent have identical filesystem layers.
The native partition-cache capacity is 64 per inference thread/queue.

Native bootstrap toolchain: vLLM 0.31.0,
Torch 2.14.0+xpu, oneAPI 2026.1.1,
oneDNN 3.13.4, C++20 and SYCL 9 for the 0.31 update.
Runtime distribution pins are in `config/experiments/b70-stack-20261010/runtime_requirements.json`.
The default `scripts/build-image.sh` builds the pinned vLLM 0.31 XPU compiler/base;
`docker/Dockerfile.vllm030-gptq-legacy` preserves the historical GPTQ base recipe.

From the extracted source, build a fresh candidate with:

```bash
python3 scripts/build_target_image.py --tag <fresh-candidate-tag> \
  --output <fresh-receipt-directory> --builder-image <pinned-builder-image> \
  --base-image <pinned-base-image> --gptq-repo <this-project> \
  --m04-artifacts <verified-Torch-2.14-M04-directory> \
  --onednn <oneDNN-3.13-install>
```

`--native-artifacts` permits only source/hash/ABI-verified native reuse.
M04 source remains in `benchmarks/experiments/exl3-shared-kv-verify`; the executed
Torch 2.14 wrapper and qualification controllers are in
`benchmarks/experiments/vllm-031-upgrade`. Native libraries must be rebuilt for
the new Torch ABI. A build alone does not approve deployment: the release
manifest links the actual gate assessment and strict service startup receipt.
No changes were pushed to 0xSero upstream.

After producing the native 0.31 / Torch 2.14 base, apply the Python overlay:

```bash
python3 benchmarks/experiments/vllm-033-upgrade/build-candidate.py \
  --base-image <your-native-0.31-image-id> \
  --wheel /path/to/vllm-0.33.0+xpu-cp38-abi3-manylinux_2_34_x86_64.whl \
  --out <fresh-build-directory> \
  --tag local/b70-qwen38-vllm:my-v033-candidate
```

Run this second command from this recipe checkout. The official final XPU wheel
is [pinned by commit](https://wheels.vllm.ai/4ba787ec9959e566d3b587a796dff07673324da5/vllm-0.33.0%2Bxpu-cp38-abi3-manylinux_2_34_x86_64.whl),
SHA256 `0e3735aa6b9ac132a32b54a948bb0de079e1b783ec5b04b184f4bdaf873eb995`.
The builder checks that wheel, the entire native source archive and all port
inputs. Necessary Python dependencies change; Torch, native libraries and the
Intel toolchain remain fixed. A successful build does not inherit this host's
qualification. Follow the [numerical, serving and operational gates](../../benchmarks/results/vllm-033-upgrade-20261108/README.md)
before creating an independent release manifest.

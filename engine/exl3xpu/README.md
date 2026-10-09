# Qualified EXL3 v3 engine source

The complete source archive preserves the upstream license. The manifest pins
upstream/base, source commit, all 74 build inputs,
native compiler/ABI identities and the actual qualified production image.
Weights, native binaries, caches and large numerical arrays are omitted.
Prior v1/v2 configuration and source snapshots remain under `config/releases`.

Extract `source.tar.gz`, or apply `migration.patch` to the recorded upstream
base. The release child and runtime parent have identical filesystem layers.
The native partition-cache capacity is 64 per inference thread/queue.

Current toolchain: vLLM 0.31.0,
Torch 2.14.0+xpu, oneAPI 2026.1.1,
oneDNN 3.13.0, C++20 and SYCL 9 for the 0.31 update.
Runtime distribution pins are in `config/experiments/vllm-031-upgrade/runtime_requirements.json`.
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

# Pinned EXL3 engine source

This directory publishes the complete tracked EXL3 source snapshot and the exact upstream-to-migration patch in the user-owned repository. It does not modify0xSero upstream. The source commit/base and SHA256 manifests identify every runtime/build input. Native binaries, model weights, caches and large measurement arrays are omitted. The compressed source is346KB; the patch157KB.

Extract `source.tar.gz` or check out the manifest upstream/base and apply `migration.patch`. The source snapshot includes the upstream license. Runtime source matches the immutable candidate6cf48999; later test-only commits isolate Dynamo compiler state and document unsupported large-prefill graph capture.

Build with `scripts/build_target_image.py --tag <fresh-tag> --output <fresh-receipt-directory> --gptq-repo <this-project> --m04-artifacts <verified-m04-directory> --onednn <oneDNN3.13-install>`. The builder/base image identities are pinned in the source script. The default native build recompiles against Torch2.13/SYCL9; `--native-artifacts` is only for source/hash/ABI-verified reuse. M04 source and its build recipe are in this project at `benchmarks/experiments/exl3-shared-kv-verify`; compiler, pinned external header revisions, library and source identities are recorded in the manifest.

A build is not release approval: run the shape, attention, quality, operation and measured serving gates. Freeze metadata only with `scripts/prepare-exl3-release-image.py`; it asserts unchanged runtime RootFS. Production selection happens after final measurements and rollback review.

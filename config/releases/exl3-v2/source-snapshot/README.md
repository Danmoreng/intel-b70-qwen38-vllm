# Qualified EXL3 v2 engine source

This directory publishes the complete tracked EXL3 source and upstream-to-release
patch in the user-owned repository. The manifest pins the upstream base, source
commit, all 74 build inputs, compiler/native identities and qualified image.
Native binaries, weights, caches and large numerical arrays are omitted. The
upstream license is preserved inside `source.tar.gz`.

Extract `source.tar.gz`, or check out the upstream/base and apply `migration.patch`.
The runtime parent and final metadata-only release child have identical filesystem
layers. The child explicitly sets the partition-cache capacity to 64. The
historical `review-candidate` directory preserves the pre-qualification build
receipt; its old candidate status is not the current release decision.

Build with `scripts/build_target_image.py --tag <fresh-tag> --output <fresh-receipt-directory>
--gptq-repo <this-project> --m04-artifacts <verified-m04-directory> --onednn <oneDNN3.13-install>`.
The source pins builder/base identities. Native recompilation uses Torch 2.13 /
SYCL 9; `--native-artifacts` requires source/hash/ABI-verified reuse. M04 source
and build recipe remain in `benchmarks/experiments/exl3-shared-kv-verify`.

A build is not release approval. The [v2 release report](../../docs/EXL3_RELEASE_V2_REPORT.md)
links the actual serving, numerical, operating and deployment gates. Use a fresh
build tag, never overwrite a qualified production or rollback alias.
The [saved v1 source and runtime](../../config/releases/exl3-v1/README.md) remain
available for immediate rollback. No changes were pushed to 0xSero upstream.

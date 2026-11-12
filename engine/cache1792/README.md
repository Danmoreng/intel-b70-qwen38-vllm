# Fine-prefix1792/128 cache release

This release retains the measured v6 EXL3/MTP3 native kernels and adds the
qualified 1792-token physical pages,128-token prefix matching and explicit
uncapped long prefill (total batch 4096). `cow_copy_fix.py` is byte-identical to
the helper used by the numerical, RAM, lifecycle, agent and full-serving gates.

`install_release.py` checks the parent runner/profile and helper hashes, then
activates the helper automatically after real KV initialization, before any
request. Profiling allocations skip installation. The helper validates the
native copy function and selects the largest covering dense alias view per
pointer. It leaves native copies, streams and matrix/model kernels unchanged.
Startup emits `B70_COMPLETE_COW_READY` with the installed helper/native hashes
and the validated 65→17 view selection. No developer endpoint or numerical
instrumentation is needed in production.

The earlier sporadic BCS GPU-memory fault remains unexplained and was also
observed with the previous profile. The operator accepted the documented
warm-cache/cold-latency tradeoff and open stability issue on 2026-11-12. This
release does not claim to fix that fault. The previous v6 release is archived
under `config/releases/exl3-v6` with its own RAM modules and rollback launcher.

Build the release layer from the repository root, with the pinned v6 parent
image already available locally:

```bash
docker build -t local/b70-qwen38-vllm:cache1792-development engine/cache1792
```

The Dockerfile retains the native libraries and applies the guarded Python
changes. Rebuilt image IDs need their own validated release manifest; the
checked-in manifest identifies the exact image deployed on this host.

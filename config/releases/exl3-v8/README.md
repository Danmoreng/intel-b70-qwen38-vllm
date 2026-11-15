# Saved EXL3 v8 release

Qualified and deployed on 2026-11-15. This snapshot retains the exact R12
image, frozen policy, unchanged tested RAM modules and deployment receipts.

[Approval, measured tradeoffs and permanent-service start/restart](../../../benchmarks/results/direct-xpu-followup-20261115/release/README.md).

- Image: `local/b70-qwen38-vllm:production-exl3-v8`
- Immutable image: `sha256:131dd9e11809de2f3136547618f130ee238f0b90cd5718567bcdd0dad5360e19`
- Runtime: vLLM 0.33.0 / Torch 2.15.0+xpu, upstream PTI fix, corrected packed M04.
- Previous production and immediate rollback: [saved v7](../exl3-v7/README.md).

`production_image.json` binds this snapshot to its own RAM manifest and
modules. `snapshot-inventory.json` protects those files. The Docker image and
model weights remain local artifacts.

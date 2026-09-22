# M12: production stack on published vLLM 0.30.0

The candidate uses the official XPU image
`vllm/vllm-openai-xpu@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d`
and the existing production Dockerfile with `BASE_IMAGE` overridden. The
resulting candidate ID is
`sha256:05e0981ea0a37ed82b309dbba3157f57c9a8144aadbf2bcb8746bfed70d95016`.
It retains Compute Runtime 26.35, IGC 2.41.5, the pinned cookbook fixes, the
local INT4 draft optimizations, and the same Q128/M04 binaries and adapter.
The official 0.30.0 image still carries XPU kernels 0.1.14.1 and Torch
2.13.0+xpu, like the production 0.29 image. Both custom libraries import and
register successfully in the candidate container; the serving run also
confirmed dispatch of both. No binary rebuild was required for this screen.

Build from the repository root:

```bash
docker build --pull=false \
  --build-arg BASE_IMAGE=vllm/vllm-openai-xpu@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d \
  -t local/b70-vllm:030-initial -f docker/Dockerfile docker
```

`run_ab.py --run-dir benchmark-results/v030-screen` prints the dry plan. Add
`--execute` for a short B/A/A/B screen against the exact production image. It
uses 4K/1K C1 and C4, two measured repetitions per arm and
one full-shape warmup. All requests are greedy like the published 2026-09-22
phase/concurrency baseline. The harness uses separate compiler caches per arm,
requires Q128 and M04 dispatch when MTP is enabled, and restores production
after success or failure. Add `--disable-mtp` to remove `--speculative-config`
from both server arms. With MTP off, the harness instead requires zero draft
tokens and no M04 dispatch. Q128 dispatch is recorded but not required: it
was inactive in both versions for this no-MTP workload. A win would still need
correctness, long-context, vision, tools and coding qualification before
promotion. The completed MTP run is documented in
[`../../runs/2026-09-22-vllm-030-short/README.md`](../../runs/2026-09-22-vllm-030-short/README.md).
The no-MTP follow-up is in
[`../../runs/2026-09-22-vllm-030-no-mtp/README.md`](../../runs/2026-09-22-vllm-030-no-mtp/README.md).

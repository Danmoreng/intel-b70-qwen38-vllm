# Reproduce the EXL3 release measurements

Use exclusive GPU access and fresh output directories. Commands below refer
to retained local v1 qualification artifacts; source data and model weights
are not bundled with the repository.

To repeat the full source matrix and QueueKit on the current permanent service:

```bash
python3 scripts/run-readme-benchmarks.py \
  --fixture-root /path/to/original/run-20260923-201101-w0.00 \
  --output-root benchmark-results/readme-new-run
```

The admission check reads the frozen policy (16 for this profile); the measured
matrix stays 20 scenarios / 70 waves / 124 requests at C1–C4. For an isolated
64K cold/warm resend, restart the service first and then run:

```bash
python3 scripts/current-profile-benchmark.py \
  --base http://127.0.0.1:8081 --container b70-qwen38-vllm \
  --expected-max-num-seqs 16 \
  --fixture-root /path/to/original/run-20260923-201101-w0.00 \
  --legacy-prefix-namespace --only prefix-64k-cold-warm \
  --output-root benchmark-results/prefix-64k-new-run --execute
```

The full paired release
controller and validated exports are in `scripts/run-exl3-final-readme.py`,
`summarize-readme-benchmarks.py` and `summarize-web-coding-benchmark.py`.
The large original fixtures/raw events remain local with 248 frozen file hashes.
The frozen v7 fixture README preserves its historical calibration instructions.
Its old `run-web-coding-campaign.py` entry point refuses an EXL3 production
service to prevent mislabeling it as GPTQ. Use the final release controller for
the corrected pair, or the standalone runner with the current service for a
single-engine repeat. Keep the six stages, 40-minute budget, 4K thinking budget
and fresh-worker warmup unchanged when comparing results.

On this installation, repeat the complete measured release campaign using the
retained qualification receipts and a fresh output directory:

```bash
python3 scripts/run-exl3-final-readme.py \
  --image-receipt benchmark-results/exl3-release-image-v1/image.json \
  --quality-review benchmark-results/exl3-optimized-quality-v2/quality-review.json \
  --operations-gate benchmark-results/exl3-optimized-operations-v2 \
  --performance-gate benchmark-results/exl3-optimized-performance-v1 \
  --fixture-root /path/to/original/run-20260923-201101-w0.00 \
  --out benchmark-results/exl3-repeat-new-run
systemctl --user start b70-qwen38-vllm.service
```

Run exclusively while the service is idle. The controller leaves workers off
after measuring; the last command restores the qualified current service.


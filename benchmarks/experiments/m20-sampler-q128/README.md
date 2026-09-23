# M20: sampler operator and Q128 profile

Run the short, restore-safe isolated probe from the repository root:

```bash
python benchmarks/experiments/m20-sampler-q128/run.py \
  benchmarks/experiments/m20-sampler-q128/runs/probe-YYYYMMDD-HHMM
```

The runner holds the shared GPU lock, verifies idle production and the exact
promoted image, stops the service once, runs both probes, then restores and
verifies the service in a `finally` block. Each probe has a 10-minute timeout.
The original run was `runs/probe-20260923-1845/`. Raw traces, telemetry and
samples remain local under ignored `runs/`. Compact results and decisions are
in the [tracked report](../../runs/2026-09-23-sampler-q128-profile/README.md).

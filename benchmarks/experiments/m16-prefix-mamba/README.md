# Long Mamba/prefix-retention probe

`long_retention.py` fits an exact 160,000-token prompt and a 190,000-token
extension, verifies that the first 160,000 token IDs are identical, and sends
the prefix followed by its warm extension. Run `--phase warm` first, restart
the unchanged production service to clear prefix cache, then run `--phase replay`
with the same output path for an identical cold extension. The live vLLM 0.30
production server does not expose the development-only `/reset_prefix_cache`
route (HTTP 404), which is why the service restart is needed.

Each request uses greedy sampling, seed 38, MTP4 and a 32-token output budget.
The runner stores only hashes, usage, timings and metric deltas; it does not
store the long synthetic prompts or generated text. The raw result is
[`long-retention-v030.json`](../../runs/2026-09-23-prefix-boundary/long-retention-v030.json).

```bash
python3 benchmarks/experiments/m16-prefix-mamba/long_retention.py \
  --phase warm --output benchmarks/runs/2026-09-23-prefix-boundary/long-retention-v030.json
systemctl --user restart qwen38.service
python3 benchmarks/experiments/m16-prefix-mamba/long_retention.py \
  --phase replay --output benchmarks/runs/2026-09-23-prefix-boundary/long-retention-v030.json
```

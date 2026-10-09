# B70 draft-only W4A16 experiments

These candidates are diagnostic, not production defaults. Target EXL3 and the
full shared target head retain their qualified paths. All runtime dense
parameters and activations are FP16; source BF16 small tensors are converted
offline once. Packed coefficients are transposed losslessly, not requantized.

Pinned source: [syvai/Qwen3.8-27B-DFlash2-W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16),
revision `4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`. `prepare_checkpoint.py`
checks the exact source SHA, 36 W4/G128 symmetric modules, all shapes, scales
and dense dtypes. Unknown runtime namespaces fail. Checkpoint-local layers0–4
map only in the loader lookup from runtime attention layers64–68.

The draft uses the existing oneDNN `_xpu_C.int4_gemm_w4a16` operator. It is
**not an EXL3-encoded draft**: the main model remains EXL3. FC, Q/K/V/O and MLP
are packed W4; Conv kernel projections, selector and norms remain FP16.
The integer encoding follows the offset-binary packing described in the
[compressed-tensors primary documentation](https://docs.vllm.ai/projects/compressed-tensors/en/latest/api/compressed_tensors/compressors/pack_quantized/helpers/).
Runtime GPU results must pass independent FP64 packed-dot probes spanning
every fused shard before a serving screen runs.

Each image change is separate:

1. `Dockerfile`: actual W4, full QKV for context; unchanged cache grouping.
2. `Dockerfile.group4`: explicit opt-in grouping for the actual 48 Mamba /
   16 full-attention / 5 draft-SWA geometry. Four layers per shared memory
   group, **not four draft tokens**. No draft marker changes. CPU replay
   preserves all69 specs exactly and checks12/4/2 groups and invalid geometry.
3. `Dockerfile.kv-only`: packed context K/V columns alias the exact QKV
   coefficients, with a small contiguous scale buffer. Normal draft-query QKV
   is unchanged; independent probes compare KV-only output to the full path.

All images, mounted source snapshots, loader receipts, raw responses, metrics
and exact rollback identities are saved by `dflash2-serving-v031/run_serving.py`.
Actual 192K input+output is a separate capacity gate; startup or finite tensor
outputs do not qualify correctness, speed or context-end sampling.

On 2026-10-09 the user explicitly permitted about166K total tokens as a
fallback if DFlash has a clear speed advantage.192K remains preferred. This
supersedes the strict192K-only acceptance rule in the original attachment;
it does not qualify any unmeasured context length.

Failures are preserved: the first preflight found root-only weight permissions;
the first GPU loader rejected the intentionally dense Conv projection because
its explicit exclusion was missing. Both were corrected before numerical
testing. Unchanged grouping rejected192K:8.92GiB required,7.94GiB available,
estimated maximum166400. That estimate is not an actual long-request result.

## Matched controls and bounded panel

`analyze_grouping.py` checks the two96K arms against the same draft SHA and
requires exact256-token greedy outputs across groups and cold/warm state,
with actual cache hits in warm waves. Eight selected actual proposal rounds
per arm are independently replayed; capture hooks are removed before timing.
Stochastic1024-token screens keep all three repeats and their acceptance.

`run_panel.py --projection full-qkv` and `--projection kv-only` use identical
three-cell frozen512-token inputs, sampling, cache salts and configured
max_num_seqs16 at192K. `analyze_panel.py` reports comparisons to the earlier two
MTP3 arms as observational, with no paired confidence claim or promotion.
It includes native request-weighted decode, aggregate client intervals,
TTFT/end-to-end guards, source hashes and exact production restoration.
K/V-only adds separate64-/4096-row projection probes across all five layers.
The analyzer also records periodic scheduler observations and a separate
per-request reservation model. Four submitted requests do not guarantee a
four-request simultaneous decode batch when KV/state capacity is constrained.

Image IDs and actual captured runtime sources are authoritative. Helpers can
evolve between diagnostic arms; an image rebuild with newer source bytes
requires its own qualification and does not inherit an old image receipt.

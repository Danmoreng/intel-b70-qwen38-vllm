# Why the 4K C1 decode rate changed

This short diagnostic used the running production vLLM 0.30.0 image with XPU
kernels 0.1.15.4, MTP4, Q128/M04, FP8 KV and the 180 W card cap. The
[content-free measurements](summary.json) include the image ID, prompt hashes
and raw-result hashes. The production configuration was not changed.

The earlier [vLLM 0.29 baseline](../2026-09-22-current-profile/summary.json)
reported 80.3 native decode tok/s at 4K input and 1,024 output tokens with
`temperature=0`, 63.3% MTP acceptance and an artificial `x`-filled coding
prompt sent to `/v1/completions`. The paired
[vLLM 0.29/0.30 A/B](../2026-09-22-vllm-030-short/README.md) measured 80.1
versus 81.3 tok/s on the same greedy prompt payloads. The current
[full source-review suite](../2026-09-23-meaningful-full/README.md) measured
67.6 tok/s and 47.5% MTP acceptance at 4K/1K using sampled chat requests.
The shorter old context points also used 128 or 512 output tokens, while the
current suite uses 1,024 throughout.

## Paired sampling check on the current image

Four 4K prompts from the full source-review run were each sent twice, once
with `temperature=0` and once with `temperature=1`. Each pair used identical
prompt bytes, seed, top-p 0.95, top-k 20, `ignore_eos=true`, disabled thinking
and exactly 1,024 output tokens. The order was greedy/sampled,
sampled/greedy, sampled/greedy, greedy/sampled. The second arm in each pair
reused 1,664 prompt tokens; the order balances this effect. Decode rate uses
post-first tokens divided by native decode seconds, excluding prefill time.

| Pair | Greedy decode | Sampled decode | Greedy MTP accepted | Sampled MTP accepted |
|---:|---:|---:|---:|---:|
| 1 | 84.72 tok/s | 65.46 tok/s | 63.4% | 44.3% |
| 2 | 76.97 tok/s | 68.82 tok/s | 55.4% | 47.9% |
| 3 | 70.98 tok/s | 67.33 tok/s | 49.3% | 46.4% |
| 4 | 74.89 tok/s | 62.72 tok/s | 53.6% | 42.0% |
| Mean | **76.89 tok/s** | **66.08 tok/s** | **55.5%** | **45.2%** |

The mean within-pair decode change was **−13.7%**. Sampling was slower in
all four pairs, and each sampled run accepted fewer draft tokens. This check
changes both sampler behavior and the resulting token sequence, which changes
MTP acceptance. It cannot assign an exact fraction of the slowdown to sampler
operator cost versus extra target-model verification.

## Replay of the archived coding prompts

The original two measured 4K coding prompts were reconstructed with the old
runner. Their SHA-256 hashes matched the archived vLLM 0.30 candidate request
payloads byte for byte. Replaying them with greedy `/v1/completions` on the
current image gave **74.53 and 84.70 tok/s** (mean **79.62 tok/s**) with
**52.9% and 63.5%** MTP acceptance. Both had zero prefix-cache hits. This
small replay overlaps the archived greedy ranges and does not show a large
engine-level C1 regression. Two requests cannot resolve a small stack change.

The evidence points to the changed workload and its lower MTP acceptance
under sampling as the main reason that the current 4K/1K table is slower than
the old greedy table. The tables describe different request distributions and
should not be used as a vLLM-version A/B comparison. Raw native metrics and
request counts remain in the ignored local `benchmark-results/diagnose-sampling4k/`
and `benchmark-results/diagnose-old-coding-prompt/` directories.

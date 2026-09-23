# vLLM 0.30 with XPU kernels 0.1.14.1 versus 0.1.15.4

On 2026-09-23, an isolated short A/B compared the current production image
`sha256:cd6562f03c8328fe60ca69269d0e4175a284859e56525950fbfc021be19d73f3`
with candidate image
`sha256:648132c9b9da4bb244d7304b956c1a9bb825be640a92755a5ffe32f2bbd679b4`.
The only intended package change was `vllm-xpu-kernels` 0.1.14.1 → 0.1.15.4.
vLLM 0.30.0+xpu, Torch 2.13.0+xpu, Q128/M04 binaries, model, MTP4, FP8 KV,
200,704-token ceiling and 180 W limit were the same. The candidate installed
the [published wheel](https://pypi.org/project/vllm-xpu-kernels/0.1.15.4/),
verified by SHA-256
`4ec262f7afdd07c62defc8286dc9f357569627b996c09b410f7972b653696fe7`.
The [v0.1.15 release notes](https://github.com/vllm-project/vllm-xpu-kernels/releases/tag/v0.1.15)
include sampling work, but changing the wheel alone does not add the separate
vLLM MRV2 sampler integration to the 0.30 release.

The four-arm candidate/control/control/candidate run took about 15 minutes,
including server starts, warmups and production restoration. Every arm used
one C4 warmup, then two C1 and two C4 measured waves. The four pinned German
Wikipedia prompts had 4,053–4,090 tokens. Sampling used `temperature=1`,
`top_p=0.95`, `top_k=20`, `max_tokens=512` and fixed per-request seeds. Prefix
cache was reset before every wave; candidate and control had separate compiler
caches. The output lengths were allowed to vary. All 40 measured requests
returned nonempty outputs of at least 128 tokens, with no reasoning content,
prefix hits or preemptions. MTP drafted and accepted tokens, and Q128/M04
dispatch was observed in all arms.

| Load | 0.1.14.1 native decode | 0.1.15.4 native decode | Change | 0.1.14.1 / 0.1.15.4 MTP acceptance |
|---|---:|---:|---:|---:|
| C1 | 59.92 tok/s | 61.71 tok/s | +2.98% | 38.95% / 40.84% |
| C4 per request | 37.50 tok/s | 37.12 tok/s | −1.01% | 43.97% / 43.11% |

C4 fully overlapped aggregate decode was 183.27 → 181.98 tok/s (−0.70%).
The C1 mean is lifted mainly by the first candidate arm's one request, which
accepted 45.7% of drafts and decoded at 66.38 tok/s. The second candidate arm
matched the second control arm almost exactly for both C1 requests and MTP
acceptance. Sampled output hashes did not consistently match across arms, as
also seen in the earlier vLLM version comparison. Four waves per load are too
few to resolve effects around 1%, and acceptance/output variation is material.

The wheel passed this serving compatibility screen, but it produced **no stable
performance improvement**. Keep 0.1.14.1 in production. The standalone
new-wheel image remains an experiment; long context, vision, tools and a
quality comparison were not run for promotion.

Evidence: [`plan.json`](plan.json), [`summary.json`](summary.json), the four
content-free arm results ([1](01-candidate.json), [2](02-control.json),
[3](03-control.json), [4](04-candidate.json)), and
[`production-restored.json`](production-restored.json). The runner and build
recipe are in [`m19-xpu-kernels-0115`](../../experiments/m19-xpu-kernels-0115/).

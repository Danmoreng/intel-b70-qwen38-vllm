# Wikipedia MTP A/B: vLLM 0.29 vs 0.30

Run date: 2026-09-23. Completed run ID: `wiki-mtp-ab-20260923`.
Wall time including server starts, warmups and production restoration:
20 minutes 42 seconds (13:33:32–13:54:14 CEST). The raw, ignored local
`benchmark-results/wiki-mtp-ab-20260923/` directory contains frozen prompts,
responses, streams, metrics and server logs. The tracked
[`summary.json`](summary.json) contains source revision links and hashes plus
aggregate measurements, without article or generated response text.

## Workload and controls

The source texts are pinned German Wikipedia revisions:
[Marie Curie](https://de.wikipedia.org/w/index.php?oldid=270535757),
[Apollo 11](https://de.wikipedia.org/w/index.php?oldid=268725520),
[Photosynthese](https://de.wikipedia.org/w/index.php?oldid=268891481), and
[Berlin](https://de.wikipedia.org/w/index.php?oldid=270716683).
Each prompt has 4053–4090 tokens in total, mostly article text, plus an instruction to
write a factual German summary of about 250–300 words. The measured chat
requests used `temperature=0.7`, `top_p=0.9`, `max_tokens=512`, a fixed
per-request seed, and `enable_thinking=false`. This was a sampling workload,
not greedy decoding. The same prompt hashes, seeds, article assignments and
server arguments were verified across all four arms.

Order: vLLM 0.30, 0.29, 0.29, 0.30. Each arm ran one C4 warmup, then four
C1 waves covering all four articles and two waves each at C2, C3 and C4.
There were 40 measured waves and 88 requests overall. The production MTP4
configuration and Q128/M04 binaries were present and both custom kernels
dispatched in every arm. Each arm used a separate compiler cache per version;
the prefix cache was reset before each wave. All measured responses were
nonempty, contained at least 128 tokens, had no reasoning content, and ended
normally or at the 512-token cap. There were no prefix-cache hits or
preemptions. Sample outputs were inspected for article-appropriate summaries.

## Results

Mean measured values by concurrency; C1 has eight waves per version, the
others four per version:

| Load | Native decode tok/s: 0.29 → 0.30 | Fully overlapped aggregate tok/s: 0.29 → 0.30 | Batch wall: 0.29 → 0.30 | MTP acceptance: 0.29 → 0.30 |
| --- | ---: | ---: | ---: | ---: |
| C1 | 67.95 → 69.02 (+1.57%) | 67.94 → 68.98 (+1.53%) | 9.95 → 9.71 s | 48.2% → 48.8% |
| C2 | 51.08 → 52.94 (+3.65%) | 112.35 → 116.12 (+3.36%) | 14.85 → 14.36 s | 45.5% → 48.1% |
| C3 | 45.61 → 45.50 (−0.26%) | 152.51 → 151.42 (−0.72%) | 18.32 → 18.44 s | 48.0% → 48.0% |
| C4 | 40.13 → 39.54 (−1.47%) | 201.45 → 196.24 (−2.58%) | 21.54 → 21.60 s | 49.9% → 49.1% |

The earlier coding-prompt MTP screen showed an 11.0% C4 aggregate decode
regression on 0.30 and a 10.4-percentage-point acceptance drop. **That large
effect did not recur** in the Wikipedia summaries: acceptance was around 50%
for both versions, with only a 0.9-point weighted C4 difference. This
supports task dependence of the MTP result. The summarization task did not
produce the especially high acceptance hypothesized before the test.

C1 varies by article. Across the two arms per version, native decode for
Marie Curie was 66.43 → 68.33 tok/s, Apollo 11 was 70.43 → 68.37,
Photosynthese was 65.42 → 64.77, and Berlin was 69.52 → 74.60. Thus the
+1.57% overall C1 mean is not a uniform gain across articles. At C4, the
four paired aggregate-decode changes ranged from −7.06% to +0.91%; the
mean regression is much smaller than in the coding workload. Batch wall time
was nearly identical while generated token counts varied naturally.

Pinned seeds ensured identical *requested* sampling conditions, not identical
outputs: only 3 of 22 outputs matched between the repeated 0.29 arms, and
0 of 22 matched between the repeated 0.30 arms. This nondeterminism and the
limited number of articles prevent a precise claim that 0.30 is faster at
C1/C2 or slower at C4 for all summarization tasks. The result does establish
that the earlier 11% C4 regression is not a constant cost of vLLM 0.30 MTP.

The runner restored the original production vLLM 0.29 image and 180 W cap;
`/health` returned HTTP 200. No production deployment was changed.

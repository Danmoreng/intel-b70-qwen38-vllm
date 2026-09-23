# vLLM 0.29 versus corrected 0.30: prefix boundary screen

On 2026-09-23, a short screen tested the report's MTP/Mamba prefix-cache
boundary hypothesis. The running engine logged KV-cache group sizes
`[1664, 1664, 1664, 1664]` and scheduler LCM **S=1664**. The arms used the
same model revision, FP8 KV, MTP4, Q128/M04 binaries, 180 W, `align`, C1 and
max batch 6656. v0.29 used its production image
`sha256:b675d81d4e7cc63fbcd6df395965ea16ec5c4704428c81118a1618185245dd5a`.
v0.30 used the reproducibly rebuilt image
`sha256:cd6562f03c8328fe60ca69269d0e4175a284859e56525950fbfc021be19d73f3`
without the incompatible local EAGLE/Mamba-drop patch. The old uncorrected
0.30 image had zero prefix hits even for ordinary long shared prefixes and
was rejected.

For each length the script sent a cold prompt, an identical resend and a
longer request whose token IDs started with the exact prompt token IDs. It
used greedy 16-token output and seed 38. Both versions used the same six
prompts (prompt and token-ID SHA-256 hashes match in the JSON records). Cold
hits and preemptions were zero in every case; MTP drafted 20 tokens per
request.

| Prompt tokens | 0.29 resend hits | 0.30 resend hits | 0.29 extension hits | 0.30 extension hits | 0.29 / 0.30 resend TTFT, s |
|---:|---:|---:|---:|---:|---:|
| 3327 = 2S−1 | 0 | 0 | 0 | 0 | 2.121 / 2.157 |
| 3328 = 2S | 0 | 0 | 0 | 1664 | 2.211 / 2.216 |
| 3329 = 2S+1 | 0 | 1664 | 0 | 1664 | 2.232 / 1.149 |
| 4991 = 3S−1 | 0 | 1664 | 0 | 1664 | 3.312 / 2.238 |
| 4992 = 3S | 0 | 0 | 1664 | 3328 | 3.369 / 3.375 |
| 4993 = 3S+1 | 0 | 3328 | 0 | 3328 | 3.357 / 1.189 |

The corrected 0.30 image improves reuse in several boundary cases and does
not worsen the exactly aligned resend relative to 0.29. The latter still has
**zero hits at 2S and 3S** on both versions; the report's expected fix is
therefore not fully realized for this production configuration. Each cell is
one request, so TTFT values are diagnostic observations rather than a
statistical performance claim. This screen does not cover the report's
150K–190K Mamba-state-retention case.

Raw results: [`v029.json`](v029.json), [`v030.json`](v030.json). Runner:
[`run.py`](../../experiments/m14-prefix-boundary/run.py).

## Mamba-state correctness spot check

The old local EAGLE/Mamba drop patch addressed a wrong-state risk, so cache
hits alone are insufficient. On the corrected v0.30 image,
[`correctness.py`](../../experiments/m14-prefix-boundary/correctness.py) asked
for five exact codes from a 22,877-token document, once with prefix caching
enabled and once with it disabled. The document hash, prompts, greedy seed
and model image were identical. All five answers and completion-token counts
matched exactly across modes and were correct. With cache enabled, each of
the four warm requests reused 19,968 tokens; with cache disabled, hits were
zero. MTP was active and neither arm preempted. Raw runs:
[`cache on`](correctness-cache-on.json),
[`cache off`](correctness-cache-off.json).

The upstream [EAGLE/Mamba checkpoint change](https://github.com/vllm-project/vllm/pull/53945)
in v0.30 shifts the written Mamba checkpoint back to the replay position; the
older [read-side drop proposal](https://github.com/vllm-project/vllm/pull/48375)
predates that path. This spot check supports keeping the upstream v0.30
behavior, while a five-question check cannot prove equivalence for every
possible request or long-context state layout.

# Intel Arc Pro B70: Qwen3.8-27B with vLLM

The qualified production profile serves **EXL3 4.00 bpw on one 32 GB Intel Arc
Pro B70 at 180 W**, using vLLM 0.33.0 and Torch 2.14.0+xpu. It exposes an
OpenAI-compatible API at `http://127.0.0.1:8081/v1`, model `Qwen3.8-27B`.

The 2026-11-12 default uses **1792-token cache pages / 128-token prefix matching**
and automatically installs the **complete dense-alias KV COW copy correction**
at every worker start. Batch 4096, max 262144 tokens, MTP3 and 12 GiB RAM ARC remain.
Warm 16K/64K prefix requests were about 10% shorter, with 68–77% shorter TTFT.
The complete 70-wave matrix was 0.28% shorter with +1.47% weighted native decode;
these small aggregate changes do not establish a universal speedup. Some cold
short requests add 0.1–0.15 s TTFT. The operator accepted these costs and the open
sporadic BCS GPU-memory fault, also previously observed with the old profile.
This release does not claim to fix that fault.
[Qualification, packaging and deployment evidence](benchmarks/results/mtp-cache-1792-20261112/README.md).

The preceding 2026-11-10 default added the **MB4-DPAS/K4 matrix patch under MTP3**,
which remains unchanged.
Against the exact previous vLLM 0.33 image, the full 70-wave / 124-request matrix
measured **+5.41% weighted native decode**, **3.36% less measured wave time**,
**3.34% less card energy**, and **−0.33% weighted prefill rate** at the same 180 W.
The user approved deployment and publication with the open follow-ups below.
This is a serving speed result, not a new coding-quality result.
[Complete evidence and limitations](benchmarks/results/mtp-matrix-20261110/README.md).
The [previous vLLM 0.33 qualification](benchmarks/results/vllm-033-upgrade-20261108/README.md)
remains the runtime and rollback basis.

Since 2026-11-07 the launcher also enables the installed vLLM diagnostics
`--enable-per-request-metrics` and `--enable-prompt-tokens-details`. These add
server timing/cache fields to completed responses (the final usage event for
SSE); they do not change the pinned image, model, numerical or sampling profile.
The preflight receipt lists these extra diagnostic flags explicitly. The agent
runtime sends model requests through the Observatory gateway, which retains the
request metadata; task metrics share its request IDs. vLLM's prompt interval is
scheduling to first token, excluding queue time; decode spans first to last
token and therefore uses N-1 output tokens. These are server intervals, not
isolated GPU-kernel timings.

The production runtime adds a **12 GiB host RAM prefix cache** and an async
scheduler that queues requests whose full input/output reservations exceed GPU
capacity. Set `B70_KV_RAM_GIB=1..12` in `.env` to change the RAM budget.
The pinned [runtime manifest](config/production_ram_kv.json) verifies the overlay
modules in addition to the immutable base image. The container uses a 23 GiB
host-memory limit, bounded 128 MiB transfer batches and no developer endpoints.
See the [measured cache capacity and limitations](benchmarks/results/kv-ram-admission/README.md).
The cache is inclusive: two alternating maximum-length histories are not guaranteed
to remain cached. The admission scheduler prevents competing large active requests
from repeatedly preempting each other. Historical candidate reports retain their
original qualification status.

Since 2026-11-08 the attested RAM profile explicitly sets
`VLLM_KV_CACHE_LAYOUT=LBNHC`. The connector's default `LBHNC` layout prevented
the frozen EXL3 oneDNN prefill and M04 verification guards from selecting their
optimized paths. At 230 W, the same-host 131K cold code prompt improved from
**141.76 to 89.22 s TTFT** with the RAM cache retained (two repeats per arm).
The launcher passes the validated layout into Docker and rejects conflicting
host overrides. That layout comparison kept the image, native guards and
checkpoint fixed; the production cap remains 180 W. See the [cause, controls, production checks and reproduction
commands](benchmarks/results/ram-kv-layout-20261108/README.md).

## Start and recover the qualified service

The local release uses the immutable image recorded in
[config/production_image.json](config/production_image.json):
`sha256:dc81825313d587ca22eaeeceb26808a8eed63cfb84b9dde22599223b8dab59d1`,
alias `local/b70-qwen38-vllm:production-exl3-v7`.
The strict launcher checks image/policy identity, native libraries, serving
arguments, checkpoint files, middleware and the 180 W cap before loading.
Compiler caches are isolated by policy and image ID.

On an installation with the qualified image and local checkpoint already
available, run from the checkout used by the user service:

```bash
cp .env.example .env                 # initial setup only; retain an existing .env
./scripts/download-model.sh
./scripts/set-power-limit.py
./scripts/install-user-service.sh
curl -fsS http://127.0.0.1:8081/v1/models
```

Check or recover an existing installation:

```bash
python3 scripts/run-server-exl3.py --check-only
python3 scripts/restore-production.py
systemctl --user status --no-pager b70-qwen38-vllm.service
journalctl --user -u b70-qwen38-vllm.service -n 60 --no-pager
```

The restore helper validates current release receipts, the service checkout
and launcher preflight before starting the service. The service uses Type=exec;
its process being active does not mean model loading has finished. Check the
API readiness endpoint. The host defaults are loopback and port 8081; the
recovery helper targets that endpoint and the standard container name.

The immediate rollback is the [saved v6 MTP3 matrix release](config/releases/exl3-v6/README.md),
with `scripts/run-server-exl3-v6-rollback.sh`, a frozen policy and archived RAM
modules. Older [v5](config/releases/exl3-v5/README.md), [v4](config/releases/exl3-v4/README.md), [v3](config/releases/exl3-v3/README.md),
[v2](config/releases/exl3-v2) and [v1](config/releases/exl3-v1/README.md) remain available.

The [EXL3 source snapshot/build instructions](engine/exl3xpu/README.md) contain
immutable upstream and native/header pins. A rebuilt image needs its own
qualification and release manifest. For a **base development build**, use a
separate tag:

```bash
B70_BASE_BUILD_IMAGE=local/b70-qwen38-vllm:base-development ./scripts/build-image.sh
```

This helper does not load the serving `.env` and rejects production/rollback
aliases before any build or download. It intentionally builds the pinned 0.31
native bootstrap. Follow the [native and Python-overlay build workflow](engine/exl3xpu/README.md)
to produce a 0.33 EXL3 candidate; a rebuild needs its own qualification.

## API defaults and example

Chat requests default to 16,384 maximum completion tokens and an 8,192-token
thinking budget. Larger thinking budgets are capped at 8,192. Reasoning effort
is `medium`; thinking and retention of thinking history are enabled by default.
`reasoning_effort: "none"` disables thinking unless explicitly overridden with
`chat_template_kwargs.enable_thinking`. Requests can set a smaller output cap.
Tools use the `qwen3_xml` parser and automatic tool choice.

```bash
curl -fsS http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B","messages":[{"role":"user","content":"Write a small JavaScript debounce function."}],"max_tokens":512,"reasoning_effort":"none"}'
```

## Current production configuration

| Setting | Applied value |
|---|---|
| Model / revision | `turboderp/Qwen3.8-27B-exl3` / `113cf7ab958054860e43fb7f3063b1af19171095` |
| Served name | `Qwen3.8-27B` |
| Weights / activations / target head | EXL3 4.00-bpw checkpoint / FP16 / 6-bpw full 248,320-row head |
| Linear dispatch | Native EXL3 SmallM through 128 rows; INT8 large-matrix prefill; true MB4 for M≤4, original 24/40/48 choices; exact K4 decoder |
| Prefill attention | Guarded oneDNN: query rows ≥64, exact active KV 4,096–262,144, query bucket 256, exact-K bucket 1; eager only |
| Verification | M04 for supported uniform q2–5 / C1–C4; direct output copy at C4; native fallback elsewhere; no duplicate KV updates |
| oneDNN partition cache | 64 exact-shape entries per inference thread/queue; completion-aware LRU eviction; oneDNN internal caches are a separate scope |
| MTP | 3 draft tokens, FP16; 65,536-row draft vocabulary; full 248,320-row target vocabulary |
| Context | 262,144 total input+output tokens |
| Admission / batch | 16 sequences; 4,096 max batched tokens; full-ISL; RAM overlay watermark 0.05 (base default 0.0) |
| Graphs / cache | FULL_DECODE_ONLY, capture sizes 1/2/4/8/12/16/24/32/40/48/56/64; FP8 KV, prefix reuse, Mamba alignment |
| Memory / power | 0.965 GPU fraction / 180 W card cap |
| Runtime | vLLM 0.33.0, Torch 2.14.0+xpu, oneAPI 2026.1.1 / SYCL 9, oneDNN 3.13.4; Intel Runtime 26.35.39758.10, IGC 2.41.5 |
| Tools / reasoning / media | qwen3_xml / qwen3; 32 images or 4 videos within the total context limit; image cap 4,194,304 pixels |

## Capacity and operating limits

- **262,144 total tokens** includes input, retained conversation/reasoning,
  media tokens and generated output. The exact **261,120 input + 1,024 output**
  boundary was qualified on pre-matrix v5. The matrix delta completed
  199,673 input + 1,024 output; its fresh 262K/RAM delta retest was interrupted
  to prioritize publication. The configured 262,144 limit is unchanged; the
  interrupted matrix-specific retest is not a pass.
- **C4 at moderate context** is qualified without preemptions. Sixteen admitted
  requests share the same cache pool. All 16 pressure-test requests completed,
  with 2 preemptions; this allows waiting/recomputation and
  does not promise sixteen simultaneous maximum contexts.
- Up to **32 images or 4 videos** per prompt, within the same context and
  memory budget. The image cap is 4,194,304 pixels. Qualification includes
  32 distinct images, long image context, video limits and recovery.
- Independent FP32 kernel comparisons and restored-state prefill, mixed-batch,
  C4-graph and long-prefill numerical probes pass at the original tolerances.
  Output distributions and generated text change across Torch/vLLM versions.
  The old same-runtime bitidentity check failed and remains recorded; the
  cross-version changes were reviewed explicitly. Finite probes do not
  guarantee identical text or all-task accuracy.
- Historical EXL3 v1 QueueKit passed **7/8 checks and 1/2 tasks**, with the
  same failure in its native-attention control. Coding tasks were not rerun
  for this release; the failed task remains documented.

Earlier [oneDNN qualification](benchmarks/results/b70-stack-20261010/FINAL_DECISION.md)
and [vLLM 0.33 upgrade evidence](benchmarks/results/vllm-033-upgrade-20261108/README.md)
retain their adverse results, failed equality receipts and finite quality scope.
They are historical evidence, not fresh matrix-release tests. No model
quantization or BF16 serving bridge is introduced here.

<!-- BEGIN CURRENT SERVING MEASUREMENTS -->
## Current source-review serving benchmark

Measured **2026-11-10**, reusing the exact pre-matrix vLLM 0.33 reference from
**2026-11-08**: **20 scenarios, 70 measured waves, 124 requests and 126,976
output tokens per arm**. All 124payloads and prompt bytes match; token counts
and finish reasons passed. Both arms: **0 preemptions**,168,000 cached prompt
tokens,16 slots,180 W,4096 prefill,LBNHC,watermark 0.05,12 GiB ARC. Sampling: temperature 1,
top-p 0.95,top-k 20,thinking disabled,`ignore_eos=true`,1,024 output tokens. These
are throughput measurements, not semantic agent scores. Warmups/model start
are excluded from measured wave totals.

| Whole measured matrix | Previous v5 | Matrix default | Change |
|---|---:|---:|---:|
| Sum of 70 wave times | 2,508.10 s | 2,423.72 s | -3.36% |
| Weighted native decode per request | 45.23 tok/s | 47.68 tok/s | +5.41% |
| Weighted aggregate decode per wave | 81.02 tok/s | 85.79 tok/s | +5.89% |
| Weighted native prefill | 1,418.36 tok/s | 1,413.64 tok/s | -0.33% |
| Output / measured wave time | 50.63 tok/s | 52.39 tok/s | +3.48% |
| Card energy | 451,528.32 J | 436,435.31 J | -3.34% |

Separate same-day controlled ABBA costs improve decode **5.99%** across five
shapes: four matched native histories and one 4K frozen-proposal completion
retaining all native draft computation. That replay is not an agent result.

### One request: context sweep

Medians across measured waves. Native decode uses N−1 post-first output tokens
per server decode second; prefill uses computed KV tokens per prefill second.

| Input budget | Waves | Actual input | Prefill tok/s | Decode tok/s | TTFT | End to end |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 5 | 479–507 | 1,709.2 | 63.58 | 0.297 s | 16.392 s |
| 2,048 | 5 | 1,992–2,047 | 2,316.3 | 64.92 | 0.878 s | 16.641 s |
| 4,096 | 5 | 4,052–4,094 | 2,197.6 | 60.51 | 1.875 s | 18.789 s |
| 8,192 | 5 | 8,167–8,186 | 2,282.0 | 59.74 | 3.602 s | 20.702 s |
| 16,384 | 5 | 16,335–16,379 | 2,160.2 | 61.53 | 7.593 s | 24.216 s |
| 32,768 | 5 | 32,704–32,762 | 1,976.6 | 57.48 | 16.598 s | 34.426 s |
| 65,536 | 3 | 65,491–65,532 | 1,668.3 | 52.37 | 39.356 s | 58.834 s |
| 131,072 | 3 | 131,034–131,070 | 1,262.3 | 44.63 | 103.958 s | 126.983 s |

### Simultaneous requests

Aggregate tok/s after all requests emitted a first token and before any
finished; separate load points, not paired scaling measurements.

| Input/output per request | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| 2,048/1,024 | 65.63 | 114.02 | 159.71 | 202.26 |
| 4,096/1,024 | 61.37 | 113.24 | 154.40 | 190.69 |
| 16,384/1,024 | 60.72 | 110.50 | 137.99 | 170.08 |

### Scope and open follow-ups

- **128K:** median native decode is effectively unchanged,44.60→44.63tok/s;
  lower observed acceptance offsets shorter rounds in this sampled panel.
- **C2/2K prefill:** −10.44% median rate against the historical reference. Two
  candidate repeats were slower,the third near baseline. Cause unresolved;
  the planned bounded ABBA was deferred for user-approved publication.
- **Long case:**199,673 input+1,024 output completed at 40.43 decode tok/s. Its
  +13.75% versus one historical repeat includes a higher acceptance rate;
  it is not an isolated kernel gain.
- No fresh eight-run natural coding-agent panel or completed matrix-specific
  262K/RAM retest is claimed. Finite operator/model gates pass. All 124 sampled
  answer hashes differ from the reference; no universal text identity or
  coding-quality improvement is established.

[All 20 scenario comparisons and raw exported counters](benchmarks/results/mtp-matrix-20261110/README.md).
<!-- END CURRENT SERVING MEASUREMENTS -->

## Pro follow-up: MTP4 and quantized DFlash experiments

These experiments keep FP16 target/draft, FP8 KV, configured max_num_seqs=16
and 180 W. They do not change the qualified MTP3 production profile above. The
[Pro work order](benchmarks/experiments/dflash2-pro-diagnostics-v031/PRO_ASSIGNMENT.md),
[initial diagnostic findings](benchmarks/results/dflash2-pro-diagnostics-v031/DECISION_A.md)
and [completed decision](benchmarks/results/dflash2-pro-optimization-v031/DECISION.md)
record scope, source/image identities, independent references and preserved failures.

A bounded MTP3/MTP4 ABBA study uses three frozen 512-output cells: 32K C1, 64K C1
and 32K prose C4, weighted 35/35/30 before measurement. Cold and warm cache states
are checked separately. MTP4 gains +0.11% cold and −1.16% warm, or −0.53% with equal
cold/warm weight. Its descriptive paired 95% interval is [−3.57%, +2.52%] from
only two AB/BA blocks. Warm C4 aggregate client decode regresses 17–19%;
mean request end-to-end latency rises 20–22%. MTP3 remains the production choice.
[All waves and paired analysis](benchmarks/results/dflash2-pro-optimization-v031/mtp4-bounded/analysis.json)
retain absolute rate corrections and the original raw source hashes.

The draft-only W4/G128 integration uses oneDNN W4A16, while the target remains
EXL3. Its logical draft parameters total approximately 1.193 GiB; the full target
head and embeddings are shared aliases. Dense draft activations, norms,
convolution projections and selector remain FP16. Source BF16 small tensors
are converted offline once, with no BF16/FP16 serving bridge.

An isolated cache grouping change from five to four layers per shared-memory
group reduces 192K padding from approximately 1.530 to 0.076 GiB. This is unrelated
to draft-token count: the draft still uses an eight-token block and seven
proposals. Required 192K pool memory falls 8.918 → 7.465 GiB. Both the full-QKV and
separate packed K/V-only candidates passed actual 190,976 input + 1,024 output
tokens, with no preemption or unexpected recomputation, plus near-limit stochastic
sampling and a one-token custom stop. 192K is a tested capacity; a larger
allocator-derived estimate is not a qualified context limit.

[Three-repeat diagnostic screens](benchmarks/results/dflash2-pro-optimization-v031/initial-screens.json)
retain all values, including large acceptance-dependent outliers. Their best
single samples cannot establish a stable speed gain. Matched-context grouping
controls and the identical C1/C4 panel are reported separately from the current
production benchmark.

The completed 512-output panel uses the same frozen inputs, seeds and cache
states. MTP3 controls use the common wide-M04 experiment image. DFlash has
one later arm per projection, so these are observational comparisons:

| Frozen cell, tok/s cold / warm | MTP3 reference mean | W4 DFlash full QKV | W4 DFlash K/V-only |
|---|---:|---:|---:|
| 32K C1 | 56.41 / 55.93 | 53.05 / 52.95 | 52.52 / 52.44 |
| 64K C1 | 47.44 / 47.45 | 44.53 / 43.23 | 49.30 / 41.57 |
| 32K prose C4, client aggregate | 32.17 / 101.01 | 25.73 / 59.12 | 25.67 / 58.60 |

MTP3 values are means of two controls. C1 is native request-weighted decode;
C4 is aggregate client decode, and all rates exclude the first output. On the
35/35/30 weighted native mix, full-QKV DFlash is −5.56% and K/V-only −4.97%.
Both fail C4 guards; warm client aggregate decode loses 41–43%. K/V-only's
cold 64K gain does not persist in the warm cell. Production therefore keeps
MTP3 at 262,144 tokens; the experimental DFlash candidates provide tested
192,000 tokens without a demonstrated overall speed advantage.

Configured max_num_seqs=16 does not guarantee sixteen simultaneous long
requests. Both DFlash panels periodically show three running and one waiting
request. Actual pool/spec ledgers document speculative-state and KV admission
pressure. Registered draft weights/buffers occupy approximately 1.256 GiB,
excluding shared target weights and separate KV/graph/work buffers.
[Complete panel metrics and admission evidence](benchmarks/results/dflash2-pro-optimization-v031/panel-comparison.json)
include throughput, latency, acceptance and the scope of the reservation model.
Static B8/V4 remains unmeasured: its conditional long-C1 cost case did not
justify implementation; this does not rule out a future C4-specific benefit.

Numerical kernel, sampler and state probes have finite
scope; graph-internal model branches and broad semantic quality are not
qualified by these captures.

<!-- BEGIN POWER COMPARISON -->
## Historical EXL3 v2 power-limit comparison: 150 / 180 / 230 / 275 W

These measurements belong to the saved EXL3 v2 release, with vLLM 0.30.0
and Torch 2.13.0+xpu. All campaigns used the same immutable image, serving
arguments, environment values, frozen prompts and sampling. The existing fresh 180 W matrix is the baseline; the
other measured variants each repeat **20 scenarios / 70 measured waves / 124 requests** plus
an isolated 64K cold/warm resend. All ordinary requests completed without
preemptions. **180 W remains the selected production limit** because it
produced the best measured performance per watt / output tokens per Wh in
this comparison. This operating point is retained for vLLM 0.31 / oneDNN 3.13.4.
The power campaigns were not repeated for this maintenance release; the
historical measurements below remain on the older stack.

These are measured hardware variants; they do not change the qualified
production policy. Power limits are not assumed actual consumption: the table
uses the card energy counter. Measured wave times exclude warmup/startup;
energy includes each measured wave's prefill, decode and request overhead.

| Power limit | Measured mean card power | Wave time | Output tok/s | Output tok/s per W | Output tok/Wh | Card energy |
|---:|---:|---:|---:|---:|---:|---:|
| 150 W | 150.1 W | 50.05 min | 42.3 | 0.282 | 1,014.5 | 125.16 Wh |
| 180 W | 180.1 W | 41.68 min | 50.8 | 0.282 | 1,015.0 | 125.10 Wh |
| 230 W | 230.1 W | 35.77 min | 59.2 | 0.257 | 925.7 | 137.16 Wh |
| 275 W | 275.0 W | 33.89 min | 62.4 | 0.227 | 817.4 | 155.34 Wh |

All efficiency columns cover the same **126,976 output tokens**, including
their prefill and request overhead. `Output tok/s per W` divides whole-wave
throughput by measured mean card power: it equals output tokens/J. `Output
tok/Wh` is output tokens divided by integrated card energy; **higher is
better**. These are not decode-only rates, and do not include whole-PC power.

For this fixed mixed workload, **180 W produced the most tokens per Wh**;
**275 W finished the measured waves fastest**. The rate tables below show
which context and concurrency points benefit from the additional power.

The observed efficiency lead over the next-best setting (150 W) is
**0.1%**. Each limit has one full campaign with the fixed
repeats, without randomized order. Small differences do not establish a
general optimum; generated trajectories and MTP acceptance can also differ.

C1 rates below are median native prefill / request-weighted decode in tok/s.

| C1 input budget | 150 W prefill / decode | 180 W prefill / decode | 230 W prefill / decode | 275 W prefill / decode |
|---:|---:|---:|---:|---:|
| 4K | 1,858.0 / 45.8 | 2,241.4 / 57.1 | 2,615.2 / 66.8 | 2,791.5 / 69.5 |
| 16K | 1,785.8 / 46.5 | 2,168.2 / 56.2 | 2,540.1 / 65.5 | 2,721.1 / 68.4 |
| 64K | 1,396.6 / 43.6 | 1,670.0 / 52.7 | 1,947.6 / 60.6 | 2,086.1 / 63.9 |
| 128K | 1,069.5 / 33.3 | 1,262.5 / 39.9 | 1,466.3 / 45.9 | 1,562.6 / 48.0 |
| 200K | 852.5 / 32.6 | 999.3 / 38.5 | 1,154.4 / 44.0 | 1,228.2 / 46.2 |

C4 rates are fully overlapped aggregate decode in tok/s, using the same
sampled-interval definition as the main serving table.

| Input per request, C4 | 150 W aggregate decode | 180 W aggregate decode | 230 W aggregate decode | 275 W aggregate decode |
|---|---:|---:|---:|---:|
| 2K | 156.1 | 190.5 | 223.5 | 231.1 |
| 4K | 146.9 | 180.1 | 204.7 | 219.0 |
| 16K | 138.2 | 165.0 | 191.5 | 202.2 |

[All 20 load points, acceptance, energy scope and temperatures](docs/EXL3_POWER_COMPARISON.md); [measurement receipts](benchmarks/results/exl3-power-20261002-with-150w/comparison.json).
<!-- END POWER COMPARISON -->

## Coding and quality results

The historical qualified v1 release completed Flappy Bird v7 in **24 min 12 s**, passed
**54/54 checks**, and reaches **90,444 input tokens**. Native weighted prefill /
decode were **1,388.7 / 53.9 tok/s**. Adaptive histories and tool work contribute
to task duration. QueueKit took **7 min 36 s**, with the failure noted above.
[Full coding results and 10K context bands](docs/EXL3_CODING_BENCHMARKS.md).

The changed-stack short panel gives PPL **3.642167** versus stored v3
**3.642175**, with mean KL against the stored BF16 reference **0.033064**
versus **0.033353**. Short JS divergence slightly worsens; quality results are
mixed. The 32K suffix PPL is **1.183419** versus stored v3 **1.190357**.
These finite probes do not establish a general accuracy or coding improvement.
The BF16 reference was evaluated from stored arrays on the CPU.
[Current quality scope and metric receipts](benchmarks/results/b70-stack-20261010/E01-numerical-review.json).

## Rollback and benchmark reproduction

The immediate rollback is the [saved qualified EXL3 v4 release](config/releases/exl3-v4/README.md).
Use `scripts/run-server-exl3-v4-rollback.sh`; it retains its own image, policy,
original RAM modules, source snapshot and compiler namespace. The saved v3, v2, v1 and GPTQ releases also remain available.
Stop the current service before using that launcher; verify the API and a real reply.

For the additional GPTQ rollback, stop `b70-qwen38-vllm.service` before switching to
`scripts/run-server-gptq-rollback.sh`; both profiles use the same API endpoint
and container name. The [independently pinned GPTQ rollback](config/releases/gptq-onednn-v2/README.md)
uses its own image, policy and compiled caches. For a persistent rollback,
follow that document's user-service instructions and verify a real request.

The [benchmark reproduction guide](docs/EXL3_BENCHMARK_REPRODUCTION.md) preserves
the frozen 20-scenario / 70-wave source matrix, QueueKit and six-stage Flappy
fixture. Run with exclusive GPU access and fresh output directories.
[Historical GPTQ comparisons](docs/EXL3_GPTQ_COMPARISON.md), the
[release report](docs/EXL3_RELEASE_REPORT.md) and
[port history](docs/EXL3_PORT_NOTES.md) retain the migration evidence.

## Sources and acknowledgements

- [Qwen model](https://huggingface.co/Qwen/Qwen3.8-27B)
- [EXL3 checkpoint](https://huggingface.co/turboderp/Qwen3.8-27B-exl3)
- [0xSero EXL3 XPU source](https://github.com/0xSero/exl3xpu)
- [vLLM](https://github.com/vllm-project/vllm)
- [Intel B70 cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)

See [NOTICE.md](NOTICE.md) and the preserved upstream license for attribution.

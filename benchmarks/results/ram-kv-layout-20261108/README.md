# Restore optimized attention with the RAM prefix cache

The production RAM overlay now explicitly selects `VLLM_KV_CACHE_LAYOUT=LBNHC`.
This fixes a layout interaction that increased cold long-context TTFT. The
immutable vLLM 0.31.0 / Torch 2.14.0+xpu image, EXL3 weights, FP16 activations,
FP8 KV, MTP3, 262,144 total-token limit, 12 GiB RAM cache and admission scheduler
are retained. Production remains at **180 W**.

## Cause and correction

In the installed vLLM, `OffloadingConnector.get_required_kvcache_layout()`
prefers `LBHNC`, a head-major layout. Without that connector the backend selects
`LBNHC`, a token-major layout. The immutable EXL3 `prefill_eligible` guard checks
tensor strides; the former layout correctly fails its supported oneDNN path.
The old RAM profile therefore used native FP8 attention. Its M04 verification
path also failed to activate.

vLLM's `resolve_kv_cache_layout()` validates an explicit environment setting
against the supported layouts and gives it priority over the connector's
preference. Selecting `LBNHC` restores the existing oneDNN and M04 paths while
retaining RAM offload. See [dispatch evidence](dispatch-evidence.txt).
No attention guard, numerical tolerance or native kernel was changed. M04's
individual contribution to the decode improvement was not isolated.

The [runtime manifest](../../../config/production_ram_kv.json) records the
required layout. Its SHA-256 is attested by the image release manifest.
`load_overlay()` rejects a missing or different layout and conflicting host
environment settings; the launcher forwards the validated value into Docker.
Simply adding a variable to `.env` would not have forwarded it through the
previous explicit Docker environment list. Historical base releases without
the RAM overlay keep their existing rollback behavior.

## Same-host controls at 230 W

One Arc Pro B70; identical image and checkpoint; frozen code prompts; C1;
greedy, seed 4242, thinking off; 512 output tokens. Each row is the median of
two requests per context. Every request had a unique cache salt and **zero GPU
and CPU prefix hits**, no preemption, one successful request and exactly 512
generated tokens in the server counters. TTFT is client time to first streamed
content; server timing independently agreed within 0.3 s. TTFT includes request
and scheduling overhead and is not an isolated prefill kernel timer.

| Profile | 32,552-input TTFT | 130,856-input TTFT | 130,856-input decode |
|---|---:|---:|---:|
| Original RAM overlay, implicit `LBHNC` | 17.378 s | 141.759 s | 49.09 tok/s |
| RAM connector removed, GPU prefix cache on | 13.907 s | 88.651 s | 56.37 tok/s |
| RAM connector and GPU prefix cache off | 13.032 s | 83.868 s | 58.10 tok/s |
| RAM overlay retained, explicit `LBNHC` | **14.142 s** | **89.217 s** | **56.90 tok/s** |

The layout correction reduced 131K TTFT by **37.1%** and increased decode by
**15.9%** relative to the original RAM overlay. The baseline and corrected RAM
arms used the same additional transfer timers and metadata-only dispatch
instrumentation. Host copy-call times can include waits for GPU compute and
must not be added to request time as independent overhead. No host-registration
or staging change is part of this fix.

Results: [summary and validation](diagnosis-summary.json),
[all 16 diagnostic cells and counter deltas](diagnosis-cells.jsonl).

## RAM restoration and production validation

In the isolated corrected-layout worker at 230 W, a marker stored only at the
beginning of a 36,907-token prompt survived a GPU-only prefix-cache reset.
The repeated request loaded **35,200 tokens / 1,382,416,384 bytes from RAM** and
returned the same marker. Cold/reload TTFT was **16.274 / 1.114 s**.
See the [canary receipt](ram-restore-canary.json). The development reset endpoint
was enabled only on that temporary loopback worker. Production has no developer
endpoints. This is a targeted reload check, not a new full 12 GiB capacity claim.

The production service was restarted through its normal strict launcher at
180 W. Docker inspection confirmed `LBNHC`, the attested RAM manifest, the same
base image and disabled developer mode. Its real dispatch logs selected
`mixed/oneDNN` and `M04` again.

Cold code prompts against that deployed service gave the following results.
The original 180 W profile has one completed sample per context; the corrected
profile has two. These are targeted checks, not a complete paired power sweep.

| Input tokens | Original TTFT (n=1) | Corrected TTFT (median, n=2) | Original decode | Corrected decode |
|---|---:|---:|---:|---:|
| 68 | 0.173 s | 0.194 s | 72.56 tok/s | 72.73 tok/s |
| 7,976 | 3.788 s | 3.597 s | 69.57 tok/s | 70.27 tok/s |
| 32,552 | 19.918 s | 16.456 s | 65.09 tok/s | 65.91 tok/s |
| 130,856 | 163.505 s | **103.407 s** | 43.52 tok/s | **49.42 tok/s** |

The two corrected 131K TTFT samples were 103.202 and 103.611 s. All eight
deployed-profile cells passed the same cold-cache and isolation checks.
See [production-180w.jsonl](production-180w.jsonl). A small image-color test
and an automatic `get_weather` tool call with parsed JSON arguments also passed.

The deployed service also passed natural GPU-eviction recovery at 180 W. A
48,059-token history was used three times, then a competing **261,926-token**
prompt forced GPU-prefix eviction. Resuming the first history produced **zero
GPU hits, 46,400 RAM hits and 1,772,355,584 completed CPU-to-GPU bytes**. All three
records stored only in that history were returned correctly, with the same
output SHA-256 as the cold request. TTFT was **26.305 s cold / 1.415 s restored**.
No cache reset or developer endpoint was used, and there was no preemption.
This demonstrates the tested history's retention, not guaranteed retention of
two alternating maximum-length histories.

Four concurrent requests with 16,027 input and 256 output tokens each also
completed with their correct markers and zero preemptions. The sampled peak
was four active requests. Receipts:
[deployment summary](production-validation.json),
[natural RAM recovery](natural-restore.json),
[C4 requests and observations](parallel-16k-c4.json),
[tool/image checks](smokes.json).

## Relation to Sergio's published benchmark

We downloaded the [cookbook at commit
7f9d650](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/tree/7f9d65020456b968416f5ecf95ff8a459aa100f7)
and used its frozen prompts and unchanged SSE timing function. The EXL3 model
revision is `113cf7ab958054860e43fb7f3063b1af19171095` in both recipes.
Sergio's published EXL3 run uses vLLM 0.26.1.dev0, 230 W, MTP3 and disabled
prefix caching. Its 131K code TTFT is about **85.36 s**. Our same-host control
with both caches disabled takes **83.87 s**: the large original gap came from
our RAM-layout integration, not evidence that vLLM 0.31 is intrinsically slower.

The original production profile achieved a median **84.50 tok/s** on the short
code prompt at 230 W versus the published **82.79 tok/s**. This is a comparison
between hosts and recipes, not a local old/new engine A/B. The initial original-
profile series contains two repeats at 230 W and one at 180 W; its final 180 W
repeat was omitted when the operator requested deploying the confirmed fix.
All 24 completed cells are retained in
[original-profile-cells.jsonl](original-profile-cells.jsonl).

The [upstream published table](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/7f9d65020456b968416f5ecf95ff8a459aa100f7/results/qwen38-27b/tensorfold-depth-20261108/summary-gpu1-230w.md)
also includes TensorFold/DFlash2, a separate engine and quantization path. We
did not run TensorFold locally. These narrow throughput cells establish neither
general coding-agent quality nor MTP-on/off byte identity. Greedy hashing of
individual outputs does not establish losslessness across engines.

## Reproduce cold production checks

With the qualified production service already running at 180 W:

```bash
git clone https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook.git sergio-cookbook
git -C sergio-cookbook checkout --detach 7f9d65020456b968416f5ecf95ff8a459aa100f7
python3 scripts/run-server-exl3.py --check-only
docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' b70-qwen38-vllm
# Select this card's hwmon directory; its numeric suffix varies between hosts.
readlink -f /sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon*
timeout 24m python3 scripts/benchmark-kv-layout.py \
  --upstream-dir sergio-cookbook/results/qwen38-27b/tensorfold-depth-20261108 \
  --hwmon /sys/class/hwmon/hwmonN --power-w 180 \
  --depths 0,8k,32k,128k --kinds code --reps 2 \
  --out benchmark-results/layout-check
```

Use a fresh output directory and an otherwise idle engine. The wrapper refuses
changed upstream files, checks the existing power cap without changing it,
assigns a fresh cache salt to each request and validates token/cache/isolation
counters. `--sampled` selects temperature 1, top-p 0.95, top-k 20 with seed 4242;
it remains a synthetic code-generation workload. The frozen prompt bytes are
not regenerated from a newer corpus. SHA-256 pins are in the script and report.

The causal controls used temporary workers from the same image: remove only
`--kv-transfer-config` for `nooffload`; additionally use
`--no-enable-prefix-caching --mamba-cache-mode none` for `noprefix`; retain all
original RAM arguments and add `VLLM_KV_CACHE_LAYOUT=LBNHC` for `tokenlayout`.
Stop the production worker before loading a temporary text engine, bind it to
loopback, and restore the service and 180 W cap afterward. The strict production
launcher deliberately refuses these experimental overrides.

The external harness is MIT-licensed, copyright 2026 SergiioB, and remains in
its upstream checkout. This recipe imports it rather than vendoring it.
Full historical matrix, numerical and capacity results elsewhere in this
repository retain their original profile and date; they were not all rerun for
this environment-only correction.

Recompute the published cold-cell checks and medians without a GPU:

```bash
python3 benchmarks/results/ram-kv-layout-20261108/verify-results.py
PYTHONPATH=scripts python3 -m unittest discover -s tests/unit
```

The host suite passes 35 tests and skips three requiring their pinned image
environment. New regressions cover a conflicting host layout, a missing or
unsupported layout in an otherwise correctly hashed manifest, budget selection,
tampered runtime/manifest refusal and the historical base rollback.

Repeat the targeted integration probes on an otherwise idle production service:

```bash
timeout 18m python3 benchmarks/results/ram-kv-layout-20261108/check-production.py \
  --out benchmark-results/layout-integration --natural-restore
timeout 5m python3 scripts/test-kv-cache-candidate.py \
  --base http://127.0.0.1:8081 --mode parallel --tokens 16000 --count 4 \
  --out benchmark-results/layout-integration/parallel-16k-c4.json
```

The near-limit competing prompt takes about five minutes at 180 W. These
commands issue synthetic requests through the existing service; they do not
restart it, enable developer mode, reset caches or change its power cap.

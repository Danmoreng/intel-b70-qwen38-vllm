# Widened M04 target verification — B70 / vLLM 0.31.0

This isolated experiment extends target attention from two through five
verification rows to **two through nine**. Eight draft tokens plus the bonus
token require nine target verification rows. The DFlash draft's own
noncausal sliding-window attention continues to use its existing backend;
M04 accelerates the causal target verification pass.

The [initial unquantized DFlash screen](../dflash2-serving-v031/README.md)
established full FP16 execution and usable 96000-token capacity. Seven draft
tokens used eight target rows and therefore the general attention fallback.
Those completed frozen requests are the comparison for widening M04 at the
same draft depth. The stored qualified MTP3 benchmark is also retained;
it is not rerun as an old baseline.

## Implementation

- Widen the strict Python and C++ shape guards to q2..9. Packed target queries
  now allow 48 through 216 query heads in multiples of 24. The packing and
  causal row formula already use the runtime row count; the mask's mathematical
  definition is unchanged.
- Retain the existing q2..5 split policy and output-copy behavior. Keep the
  production C1..C4 dispatch gate, FP16/FP8 layout, scalar scale, causal,
  full-window and feature checks. Unsupported semantics still use native
  attention. No KV cache writes move into M04.
- For new q6..9 sizes, measure packed-Q tiles 8/16 and KV splits 1/4/8/16/32.
  Tile 8 performs best in this screen. Context-dependent split choices are
  based on the actual GPU `seqused_k` values, not the worst-case CPU bound
  captured in a decode graph.
- The operator accepts an additional default-false `adaptive_splits` flag;
  ordinary callers retain the fixed policy. Its new path uses one stream-ordered
  GPU metadata kernel and the upstream forward/reducer's existing
  `splits_per_seq` support. Scratch reserves 32 splits, while each sequence
  consumes its selected count. Counts respect the existing short-length
  single-split heuristic and avoid empty trailing partitions. Zero-length
  capture padding cannot cause division by zero.
- New-size policy: up to 8192 actual KV tokens, 8 splits at C1 or 4 at C2–C4;
  through 49152, 16 splits, except C4/q9 uses 8; above that, 32 splits,
  except q9 uses 16. Split counts are adjusted for the actual number of
  nonempty 64-token tiles. This is a measured bounded policy, not a claim
  that every workload has a globally optimal setting.

Target and DFlash weights/activations execute in FP16. The previous DFlash
residual fix and its FP32 internal convolution accumulation remain in place.
This experiment adds no BF16 bridge and creates no quantized draft. Temporary
unquantized serving uses 96000 tokens; the eventual requirement remains
at least 192000.

## Build and verification

`build.py` archives the exact qualified kernel commit
`6d92b1bfbf32767ecda8e819613eb151e70030ad` and SYCL-TLA commit
`87f6850680a580654b9ea2c80dbc01aeb36ad231`. It uses the same pinned Torch
2.14/oneAPI 2026.1.1 builder and C++20 flags as the qualified production
artifact, with no GPU access. Existing upstream checkouts are not modified.

The adaptive candidate image is
`9d276b4595a7ddbd7293be19ee277f91210d8d3c9af537f69d988c708838417b`;
its M04 library SHA-256 is
`56cd3cf8d79e7fe0b5fc7043fad8f50b8084a62c62c8dd321be902b8dea73a4b`.
The child profile uses that exact checksum. The source-guarded wrapper is
installed only in the experimental image. Production remains the qualified
`194a8786f120...` image and 262144-token MTP3 profile.

### Draft-length compile cache fix

The first eight-draft-token serving attempts exposed a separate vLLM 0.31.0
cache bug: `SpeculativeConfig.compute_hash()` did not include
`num_speculative_tokens`. Reusing the image's persistent cache after the
seven-token screen loaded a CandidateSelector artifact expecting
`[16,7,5120]`/stride 35840 for `[16,8,5120]`/stride 40960. Startup failed before
timing. This was not a numerical failure of the nine-row M04 kernel.

`patch_cache.py` guards the installed source SHA-256 and adds draft length to
the speculative configuration hash. This isolates the selector, convolution
and other graph artifacts together. It does not change model computation,
clear the cache or disable compilation. The child `Dockerfile.cache` carries
the same M04 library and FP16 residual fix. `test_cache.py` proves stable,
distinct hashes for depths 3/4/7/8 for both DFlash and MTP without GPU access.
Failed campaign receipts and the original log fingerprint are retained.
The cache-fixed serving image is
`b0b9244420a2586f8328b14aeff21d94673632ad7342032a285f3b877f2de49b`;
its patched speculative configuration source SHA-256 is
`25b94e6aefae4bd257a57281b65c6b169595b22034fa3f5b9ae17394eb015a36`.

GPU micro gates passed at the original `rtol=0.01, atol=0.003`:

- q2..9, C1/C4, shuffled interleaved FP8 KV pages of 64/1600/1664 tokens,
  shortest contexts and page boundaries, native and independent FP32 reference.
- Future-token poison confirms the first row cannot attend to later rows.
- 46 graph checks include changing Q, page order and GPU lengths, plus
  heterogeneous C4 lengths crossing all adaptive policy thresholds.
- All ten fixed tile/split combinations and the adaptive policy pass native
  comparisons at 4K/32K/64K/96K. Unsupported query sizes, concurrency and
  attention semantics retain native fallback.
- The DFlash nine-token grouped convolution separately passes C1/C4 strided
  input and request-boundary FP32-reference checks, compile and graph replay.

The micro receipt contains 604 checks, 46 graphs, 20 fallback checks and
32 timing workloads. These kernel gates and timings do not alone qualify a
production serving profile or semantic long-context quality.

Selected C1 median attention times from the adaptive candidate:

| Verification rows | KV length | Native | Adaptive M04 |
| --- | ---: | ---: | ---: |
| 8 (7 draft tokens) | 4096 | 0.282 ms | 0.182 ms |
| 8 | 32768 | 0.839 ms | 0.630 ms |
| 8 | 65536 | 1.556 ms | 1.081 ms |
| 9 (8 draft tokens) | 4096 | 0.278 ms | 0.182 ms |
| 9 | 32768 | 0.951 ms | 0.637 ms |
| 9 | 65536 | 1.767 ms | 1.155 ms |

Each measurement includes query packing, scratch allocation, the metadata
kernel for the adaptive arm, attention and output-copy work in the measured
region. Nine randomized paired repetitions follow separate warmups. Kernel
latency reductions must be assessed separately from serving tokens/second.

## Frozen serving screen (2026-10-09)

| Arm | Serving limit | 4K input | 32K input | 64K input |
| --- | ---: | ---: | ---: | ---: |
| Stored qualified MTP3 | 262144 | 54.71 tok/s | 55.19 tok/s | 48.29 tok/s |
| MTP4, M04 + cache fix | 262144 | 57.07 tok/s | 59.15 tok/s | 52.51 tok/s |
| DFlash4, M04 + cache fix | 96000 | 56.33 tok/s | 51.63 tok/s | 44.42 tok/s |
| DFlash7, native target attention | 96000 | 57.56 tok/s | 48.37 tok/s | 41.07 tok/s |
| DFlash7, widened M04 | 96000 | 53.50 tok/s | 52.75 tok/s | 45.13 tok/s |
| DFlash8, widened M04 + cache fix | 96000 | 53.20 tok/s | 51.66 tok/s | 43.74 tok/s |

These are C1 native decode medians, three matching frozen measured requests per
size, each with 1024 output tokens and a separate warmup. The exact prompts,
sampling, seeds and token counts match the stored qualified MTP3 run; only its
first three repeats are used here. The production README's five-repeat medians
are a different summary. No old GPU baseline or power comparison is repeated.

At depth 7, widening M04 improves 32K/64K decode by 9.1%/9.9% relative to the
native-attention DFlash7 arm. The 4K result is lower. Stochastic acceptance and
output paths differ despite identical request seeds: weighted draft acceptance
at 4K is 27.7% with M04 versus 31.9% with native attention. Kernel micro latency
is lower, but that does not ensure higher serving throughput on every request.
Counter-valid whole-overlap round-equivalent medians for DFlash7 are
55.22/63.25/72.55 ms with M04 versus 56.32/66.44/78.20 ms natively. Those metrics
include host, draft and target work; they are not GPU-only iteration timings.

MTP4 retains the full 262144-token startup configuration, strict 409-module
target/MTP inventory and nonzero acceptance. It passes the cold/warm prefix
fixtures and improves these matching three-repeat medians by 4.3%/7.2%/8.7%
over stored MTP3. This screen does not promote it to production.

An additional exploratory MTP5..8 scout uses only the frozen 32K C1 scenario,
three measured
repeats plus a warmup at each depth. It first attempts 262144 tokens; a capacity
rejection may fall back to the authorized 192000-token minimum. Each depth
passes its own actual-worker, inventory, nonzero-acceptance and cold/warm
prefix gates, and restores the qualified production service before the next
depth. It is auxiliary reference data, not a prerequisite for DFlash/M04
acceptance. No higher-MTP 4K/64K followup is performed and no configuration
is promoted. This is a bounded search on these fixtures, not a global optimum claim.

| Auxiliary MTP scout | Configured limit | 32K decode median |
| --- | ---: | ---: |
| 5 draft tokens | 262144 | 55.95 tok/s |
| 6 draft tokens | 262144 | 59.56 tok/s |
| 7 draft tokens | 262144 | 59.54 tok/s |
| 8 draft tokens | 262144 | 54.46 tok/s |

The near tie at depths 4/6/7 does not establish a higher-depth serving winner.
These auxiliary results are kept separately from the DFlash acceptance evidence.

DFlash8 attests nine-row M04 dispatch during real graph replay. It passes the
cold/warm retrieval fixtures and a capacity-only request of 94976 input plus
1024 output tokens, without preemption. All target/draft execution remains
FP16. Neither these fixtures nor the forced-length speed outputs establish
full production semantic, multimodal, abort or concurrency qualification.

The 96000-token limit is temporary for the unquantized draft. The profiled
available KV memory is 5.86 GiB for both widened DFlash7 and DFlash8; their
reported cache token capacities are 107345 and 104188. These are allocation
estimates, not validated maximum contexts. Draft length affects speculative
state/cache reservations as well as computation. Quantizing weights alone
must not be assumed to satisfy the eventual 192000-token requirement.

### Which caches are shared?

Verified speculative tokens use the ordinary target cache for both MTP and
DFlash. That is separate from the drafter's attention state. Our Qwen MTP
also has its own full-attention predictor layer and its own K/V values;
sharing vLLM's cache allocator does not make layer contents interchangeable.
DFlash derives context K/V from target hidden states using its own projections,
then writes them to its draft-layer caches. They cannot alias the target's K/V.

The pinned DFlash2 checkpoint has five sliding-attention layers with a
2048-token window. It does not need a second complete long-context copy of
every target layer. Actual allocator reservations can exceed the pure active
window size through grouping, padding, prefix caching and speculative state.
The actual worker already shares the target embedding and full output head.
The main added fixed allocation is the 3.5845-GiB FP16 draft weights: measured
model loading grows from 16.25 GiB for target plus MTP to 19.53 GiB for target
plus DFlash. Activation, graph and cache allocations account for additional
capacity effects, and are distinct from weight quantization savings.

## Reproduce

```sh
python benchmarks/experiments/m04-wide-verification-v031/build.py --name build-adaptive
docker build -t local/b70-m04-wide:v031-adaptive-probe benchmarks/experiments/m04-wide-verification-v031
python benchmarks/experiments/m04-wide-verification-v031/run_gpu.py --image local/b70-m04-wide:v031-adaptive-probe --name micro-adaptive
python benchmarks/experiments/dflash2-serving-v031/run_serving.py --name wide-m04-k7 --depth 7 --contexts 96000 --benchmark --image local/b70-m04-wide:v031-adaptive-probe --wide-m04
docker build -f benchmarks/experiments/m04-wide-verification-v031/Dockerfile.cache --build-arg SPECULATIVE_SOURCE_SHA=88cec427b52bb1cb56842f9034a4d49a915d05302b92b0249227feaf4d81f3e1 -t local/b70-m04-wide:v031-cache-fixed benchmarks/experiments/m04-wide-verification-v031
docker run --rm --entrypoint python -v "$PWD/benchmarks/experiments/m04-wide-verification-v031:/experiment:ro" local/b70-m04-wide:v031-cache-fixed /experiment/test_cache.py
python benchmarks/experiments/dflash2-serving-v031/run_serving.py --name wide-m04-k8-cache-fixed --depth 8 --contexts 96000 --benchmark --capacity-probe --image local/b70-m04-wide:v031-cache-fixed --wide-m04
python benchmarks/experiments/dflash2-serving-v031/run_serving.py --name wide-m04-k4-cache-fixed --depth 4 --contexts 96000 --benchmark --image local/b70-m04-wide:v031-cache-fixed --wide-m04
python benchmarks/experiments/dflash2-serving-v031/run_serving.py --name wide-m04-mtp4-cache-fixed --method mtp --depth 4 --contexts 262144 --benchmark --image local/b70-m04-wide:v031-cache-fixed
```

Output directories must be fresh. GPU controllers hold the shared benchmark
lock and restore the exact qualified service on exit. `wide_inspect.py`
attests the real DFlash worker, widened wrapper, library checksum and operator
schema before timing. Frozen serving screens use the same requests, sampling,
1024-token output length and native phase timers as the initial screen.
Historical power comparisons are retained and not repeated.

[Public kernel and serving receipts](../../results/m04-wide-verification-v031/comparison.json)
include matching frozen request hashes, raw native timings, actual-worker
attestations, numerical/graph gates, build fingerprints and the source-guarded
cache fix. Production remains the qualified MTP3 image.

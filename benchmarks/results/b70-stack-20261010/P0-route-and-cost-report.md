# P0 route and cost report

Baseline immutable image `194a8786...`; MTP3, FP16, FP8 KV, 262144 max context,
180 W. Production was restored to the identical qualified image after diagnostics.
Raw receipt: `benchmark-results/b70-stack-20261010/P0-initial/campaign.json`.

Actual pure decode forms: source64K C1 uses T4/pad4/FULL. Prose32K C4
uses T16/pad16/FULL when all requests remain active; tails T12/T8/T4 are
retained. The eight-cycle C4 trace was explicitly restricted to four active
requests. Warm-prefix cache hits were observed but Mamba-aligned suffix
prefills remain; warm does not imply zero prefill.

| Window | Median round span ms | Target EXL3 GEMM ms | Target attention mainloop ms | Target reduction ms | GDN norm ms | QKV/RoPE ms |
|---|---:|---:|---:|---:|---:|---:|
| source-64k | 54.31 | 29.72 | 8.86 | 0.675 | 0.210 | 0.054 |
| prose-32768 | 76.70 | 33.40 | 19.99 | 0.756 | 0.247 | 0.060 |

Kernel values are sums divided by eight trace rounds, not disjoint critical-path
fractions. Event medians include enqueue/wait gaps and are not additive.
Control waves are only 256-output diagnostic controls, not final performance
benchmarks. Instrumented waves cannot establish a speedup.

## Conditional candidate decisions

E02/E03: `NO_MATERIAL_GAIN_PROFILE_GATE` for this campaign. Actual compiled
GDN norm kernel `triton_per_fused__to_copy_add_linear_mean_mul_pow_rsqrt_silu_view_0`
is already fused (384 calls per eight target rounds). Existing fused Triton
QKV/RoPE spans both target and MTP (152 calls). Even eliminating both
completely bounds saved device work around 0.26–0.31 ms per round; arbitrary
new SYCL kernels cannot realize complete elimination. Upstream eager headlines
are not evidence against these compiled kernels. Plan section 3 explicitly
allows skipping negligible norm opportunities; no graph exclusion/copies added.

E04a: bounded small microkernel test, not a presumed 3% production winner.
Reduction totals 0.68–0.76 ms per round; tiling may reduce part of that.
Only reduction grid/head-dimension loop is ported, preserving original
unnormalized partial-output contract, exp2 statistics, split policy and mask.

E04b: still requires attribution. All target copy kernels together cost only
0.26–0.30 ms/round, including copies unrelated to M04; do not attribute all to
output materialization. Direct layout remains conditional, not automatically
required because a mapping is possible.

E05: EXL3 GEMM dominates, but earlier MB4, vector-M4, split-K sweeps and full
fusion were negative. GDN norm + one HadIn boundary has a small upper bound;
inspect transform alignment and live tensors before accepting another fusion.
E06: no host-idle conclusion from subtracting kernel sums from wall spans.
Require a real graph-mode timeline and an avoidable operation first.

## Source pin correction

The directory named `b70-xpu-kernels-0.1.15.4-ref` is HEAD `ddf336d...` and is
not the qualified M04 header base. The actual separate M04 base is
`Local-AI-B70/qwen38/optimization/native-prefill-d-180w/kernel-source`, clean
HEAD `6d92b1bfbf32767ecda8e819613eb151e70030ad`, as required by the plan.
E04a asserts this pin. It must not import newer normalized-LSE reduction code:
our base combines unnormalized partial outputs and exp-sum/max scratch.

# vLLM 0.33 XPU qualification

This experiment ports the qualified 0.31 EXL3 profile to the final official
0.33.0 XPU wheel. It does not approve a production release. See the eventual
qualification report for measurements and the deployment decision.

The final wheel is
[`vllm-0.33.0+xpu`](https://wheels.vllm.ai/4ba787ec9959e566d3b587a796dff07673324da5/vllm-0.33.0%2Bxpu-cp38-abi3-manylinux_2_34_x86_64.whl),
SHA256 `0e3735aa6b9ac132a32b54a948bb0de079e1b783ec5b04b184f4bdaf873eb995`.
The upstream release page uses an rc3 tag despite final wheel/package metadata;
see [upstream issue 65662](https://github.com/vllm-project/vllm/issues/65662).

`Dockerfile.pinned` extends the immutable qualified 0.31 image and installs
the wheel plus explicit necessary Python dependency versions with `--no-deps`.
This preserves Torch 2.14, Triton 3.8, XPU kernels 0.1.15.4, oneAPI 2026.1.1,
the Intel userspace driver, oneDNN 3.13.4 and the EXL3/M04 native binaries.
An ordinary dependency-resolving pip install was excluded because it downgraded
the Intel runtime according to inherited Torch dependency metadata. Existing
metadata conflicts are not a claim that `pip check` passes.

`port-source.py` requires a fresh extraction of `engine/exl3xpu/source.tar.gz`
and verifies the original six input files and the entire official wheel. It
ports the Python adapters, pins 17 reviewed runtime source boundaries (including the patched MTP module and
its local policy helper) and
applies the checkpoint-format MTP policy. The native C++/SYCL math is unchanged.

Retained: EXL3 linears, pruned draft vocabulary, guarded exact-length oneDNN
prefill, M04 verification, residual GDN host-row selection, explicit LBNHC,
completion reservation and ARC GPU-prefix access tracking. The old shared-mmap
selection override is removed because 0.33 implements XPU allocation itself.
Both native H2D load queues retain independent 128 MiB staging batches and
stream drains, protecting the pageable fallback if native host registration
fails. The legacy migration trace and sparse retirement overrides stay disabled.

Upstream0.33 adds GDN backend selection; auto selects the same fused whole-layer
SYCL operation already called by0.31. Changed metadata and checkpoint planning
still require validation; unchanged binary versions alone do not establish
numerical equivalence. `run-candidate.py` exercises original-tolerance native-kernel gates,
real-state replay, startup, the frozen serving matrix and operations.
`run-quality.py` records fresh paired teacher-forced distributions for both
versions (16 short passages and four long cases with32K,102K,184K and262K prefixes); `compare-quality.py` measures finite-panel NLL, perplexity, KL and
top-token agreement. These measurements do not establish general coding quality
or identical generated text.

The serving comparison uses the same 124 frozen requests in 70 measured waves,
180 W, FP16 activations, FP8 KV, MTP3, LBNHC, 262144 maximum context, sequences16,
batch4096, decode graphs and a 12 GiB RAM cache. The established source-review
fixtures use temperature1.0, top-p0.95, top-k20 and fixed per-request seeds.
They are distinct from a short greedy maximum-throughput benchmark.
`compare-serving.py` checks all prompt bytes and payloads, accounting, finish
reasons and preemptions before comparing client/server timing, energy and draft
acceptance. Changes beyond 5% per-cell or 2% weighted throughput trigger further
investigation; they are not automatic release approval criteria.

All GPU controllers require exclusive use, stop the current user service and
restore its immutable image in `finally`. Do not start a controller while an
independent benchmark is active. The local 23:00 Europe/Berlin work limit still
applies; controller time guards do not replace planning a safe completion.

Reproduce the candidate with an existing qualified native base and the downloaded
SHA-verified wheel:

```sh
python3 benchmarks/experiments/vllm-033-upgrade/build-candidate.py \
  --wheel /path/to/vllm-0.33.0+xpu-cp38-abi3-manylinux_2_34_x86_64.whl \
  --out /path/to/fresh-build-directory \
  --tag local/b70-qwen38-vllm:my-v033-experiment
```

The builder refuses existing image tags. It builds only a development candidate;
production image tags and qualification metadata remain separate.

For an independently rebuilt native0.31 base, pass `--base-image <your-base-id>`.
Its native artifact verifier and runtime ABI must pass; the new image needs its
own complete qualification. The local comparison uses the exact immutable
previous production base, rather than substituting a rebuild.

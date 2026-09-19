# M10: mixed INT4/FP16 GDN input projection

Implemented and screened on 2026-09-19. **Rejected at the operator gate; production unchanged.**

Qwen3.8-27B uses an INT4/G128 QKVZ projection (5120 → 16384) and a separate FP16 B/A projection (5120 → 96). Concatenating them into one ordinary INT4 or FP16 GEMM would change precision or memory usage. This candidate instead executes both original formats in one SYCL kernel. Each subgroup handles an output column, coalesces packed K loads, reduces in FP32, and emits separate contiguous FP16 outputs. Only TP1, FP16 input and M=1..5 are implemented; no new quantization, sampler, MTP-depth, cache or attention change.

## Small-test results

| Rows M | Native pair, graph ms | Fused graph ms | Latency increase |
|---:|---:|---:|---:|
| 1 | 0.0944 | 0.1817 | +92.4% |
| 2 | 0.0945 | 0.1937 | +105.0% |
| 3 | 0.0948 | 0.2090 | +120.5% |
| 4 | 0.0948 | 0.2403 | +153.5% |
| 5 | 0.0953 | 0.2642 | +177.4% |

Values are medians across the per-layer medians for checkpoint layers 0, 32 and 62. The baseline is the installed `_xpu_C.int4_gemm_w4a16` plus FP16 `torch.nn.functional.linear`, with the same weights and hidden-state tensors. Each case has eight warmups and 31 randomized alternating blocks; eager and captured-XPU-graph device/wall timings are retained separately. The table reports graph device time, not full-model tokens/s.

All **15/15** layer/batch cases pass the predefined finite/allclose and relative-L2 checks (atol/rtol 0.002; relative L2 < 0.001). This is tolerance-based numerical agreement, not bitwise equivalence or a task-quality test. Weights are real checkpoint tensors; hidden states are seeded synthetic inputs rather than captured activations.

The fused kernel reduces two launches to one, but its scalar unpack/reduction implementation loses substantially more time than it saves relative to the optimized native backend. That mechanism is the likely explanation, not a separately measured per-instruction breakdown. This rejects this implementation; it does not prove that every native mixed-precision fusion is unprofitable.

## Decision and next gate

The prespecified serving gate required numerical agreement and at least 3% speedup for both graph M1 and M5 in every sampled layer. Actual graph speedups are negative in all sampled cases. Therefore **no candidate container was activated for serving**, no 8K/16K context comparison was launched, and no coding long run was started. Running those after this large operator regression would not meet the user’s requested staged testing policy. No end-to-end gain, unchanged 196K capacity or full-model quality qualification is claimed for this candidate.

For a future implementation that passes the operator gate: first test identical frozen requests at 8192/16384 tokens (then 32768/65536 if useful), with unchanged production settings, excluded warmup, a cold-cache boundary, paired/alternating arms and repeated native prefill/decode measurements. Verify MTP acceptance, actual fused dispatch, memory capacity and task correctness before proposing any long run.

## Reproduction and evidence

`build.sh` builds only the isolated operator library using the existing oneAPI 2026.0 compiler and the pinned production image. `operator-study.py` obtains the shared GPU lock, verifies idle production and 180 W, stops the model temporarily, runs the GPU-only probe and restores the unchanged production service. Its systemd launcher used ExecStopPost recovery as a second restoration attempt.

Local run: `/home/sebastian/LocalLLM/intel-b70-qwen38-vllm/benchmarks/experiments/m10-gdn-projection/runs/operator-20260919-183603`. `manifest.json` records pins and hashes; `correctness.json`, `timings.json`, `assessment.json`, `decision.json` and `production-restored.json` record evidence. Raw timing samples are excluded from git through the existing runs/ ignore rule. The measured library is an experimental operator artifact, not a production vLLM patch.

Upstream context: [installed GDN implementation and projection structure](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py), [dual-stream discussion](https://github.com/vllm-project/vllm/issues/32828). Neither is a measured mixed-precision fusion win on this workload.

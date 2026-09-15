# M03 decision

Status: retain the existing oneDNN `_xpu_C.int4_gemm_w4a16` backend.

The first native challenger kept the exact packed GPTQ W4/G128 coefficients
and FP16 activations, decoded each weight once, and reused it across M=1..5.
All seven numerical comparisons passed. The gate/up family nevertheless took
1.167--1.299 ms versus 0.185--0.188 ms for oneDNN (84--86% slower). QKV and
down projections were 92--97% slower. No serving integration was attempted.

The measured oneDNN gate/up throughput already reads roughly 89 MiB of packed
weights and scales in about 0.19 ms, leaving no evidence-backed case for a
second custom small-M implementation. Large-M behavior and production remain
unchanged.

Rollback: none required; the challenger was an isolated operator probe.

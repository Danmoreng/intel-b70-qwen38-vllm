# Final repeated103K serving comparison

Four fresh workers use one immutable6cf48999 image in native/optimized/optimized/native order; MTP3 stays fixed. Each generates512 tokens cold then warm. Native means43.98 cold/43.91 warm tok/s; optimized49.24/49.24 (+11.95%/+12.14%). Both cold waves have zero cached tokens; all warm waves reuse100800 of102752 tokens. All requests complete, without preemption or accounting failures.

This compares the complete guarded attention pipeline, including exact-K oneDNN prefill and M04 verification. Output histories differ from token8 across variants; main/draft acceptance can therefore differ. It is repeated end-to-end evidence, not an isolated kernel timing or bit-exact text comparison. The separate earlier M04-only actual-serving proof retains its own14%/identical512-token scope. The final70-wave frozen-source matrix is an independent release comparison with pinned GPTQ numbers.

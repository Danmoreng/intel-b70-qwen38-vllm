# Final frozen-runtime quality and brief profile

Existing BF16 arrays were integrity-checked and reused; no BF16 inference repeated. Natural short-panel optimized/native-control272 arrays are bit-identical. PPL3.646722/KL0.0324807 preserves the observed advantage over the pinned GPTQ3.801927/0.086369. Matching capture uses128-row measurement head batches; this is recorded measurement instrumentation, not a production head change. Historical captures with different batching are not an isolated optimization comparison.

Four engineered long prefixes32K/100K/180K/262K are untruncated;512 suffix positions give PPL1.208579 versus BF161.210748, meanKL0.00117770 over32 full distributions and32/32 top1 agreement. These repeated code/prose prefixes test precision drift, not general long-document or agent quality. The candidate summary over all prompt positions is not the suffix quality metric.

Generated controls cover20 requests/2560 tokens each atC1/C4 cold/warm. Every optimized greedy token has zero gap from the reported maximum. Nine histories differ from the native control; first-difference margins and four shared-history one-token probes are preserved in raw data. Do not claim bit-identical text or blanket negligible ties. Functional coding is separately measured after these gates. No new numerical pass threshold was introduced after viewing results.

The final kernel check is deliberately bounded: three128-token event waves and two eight-cycle pure-decode traces. Instrumentation perturbs timing. Captured target/draft graphs remain opaque to individual-kernel trace decomposition, so no invented graph-internal speed attribution or additional speculative kernel tuning follows. M04 route capture/replay is observed; end-to-end ABBA is separate.

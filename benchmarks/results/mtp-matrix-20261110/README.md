# MTP3 MB4/K4 matrix release — 2026-11-10

The operator approved deployment/publication after the complete serving sweep, with known C2 prefill and remaining agent/operational delta checks deferred. This does not claim every originally planned gate passed.

- [Decision and limits](decision.json)
- [All 20 scenario comparisons](comparison.json) and [absolute scenario summaries](serving-summary.json)
- [All 70reference wave counters](wave-metrics-A.json) and [all 70candidate counters](wave-metrics-B.json)
- [Controlled costs](controlled-cost-gate.json) and [retained prefill check](prefill-check.json)
- [A/self](model-self.json), [A/B full](model-full.json), [A/B eager](model-eager.json), [independent operator checks](operator-numerics.json)
- [Public/source byte mapping](PUBLICATION_MANIFEST.json)
- [Strict production start and smoke receipt](promotion.json)
- [Native source, patch and build](../../../engine/exl3xpu/mb4-k4/README.md)

The full matrix uses the exact November 8 image reference,124 byte-identical prompts/payloads,180 W,16 slots,4096 prefill,LBNHC,watermark 0.05,12 GiB ARC. Sampling T1/top_p.95/top_k20,thinkingdisabled,1024forced outputs. Native decode uses N−1 post-first tokens. Per-request and aggregate-wave decode have different denominators. Measured wave sums exclude model startup/warmups. This is source-review serving throughput, not a new natural coding-agent study.

Totals: native decode **+5.412%**, measured wave time **−3.364%**, card energy **−3.343%**, weighted prefill rate **−0.332%**. Same180 W cap; all 124requests complete, matching token counts,zero preemptions. All 124 sampled answer hashes differ; no universal byteidentity is claimed.

Open: C2/2K prefill rate −10.44% in two slower and one baseline-like repetition; cause unresolved. The boundedABBA and eight new genuine agent runs are prepared, not executed. The fresh262144/RAMdelta retest was interrupted for publication, not failed numerics and not a passed gate. Prior0.33 baseline capacity/RAM proofs remain historical. Currentmatrix completed199,673 input+1,024 output; configured limit 262,144 is unchanged.

The released image inherits the measured runtime and adds only compiled source, the preflight source-root correction and metadata. No forward/sampler/native execution file is changed. The deployed identity and inference-payload attestation are in promotion.json.

The protocol uses `scripts/current-profile-benchmark.py --execute --expected-max-num-seqs 16`, the unchanged public corpus/scenarios, original frozen fixtures via `--fixture-root`, and `--legacy-prefix-namespace`. Source script hashes and124payload comparisons are in decision.json. Raw prompts/streams and numerical tensors remain local; public wave counters contain no prompt/answer text. Original result hashes are mapped in PUBLICATION_MANIFEST.json.

[Immediate pre-matrix0.33 rollback](../../../config/releases/exl3-v5/README.md).

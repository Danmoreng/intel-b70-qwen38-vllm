# M05 INT2 draft-head offline gate

This experiment evaluates a full-vocabulary INT2/G128 shortlist with eight
candidates per 64-token vocabulary block and exact FP16 reranking. The target
head and target sampler are untouched. The production baseline is the existing
INT4/G128 draft head, not BF16.

The first stage is a kernel-capability and synthetic-input screen. A candidate
must then repeat the recall and timing study on captured local draft hidden
states before any serving integration is considered. Storage savings alone are
not a promotion criterion.

## Result: rejected

The Xe2 port was tested against the real local FP16 head
`[248320,5120]` and 16 distinct draft hidden states captured from an eager
MTP4 request. All INT2 variants (eight candidates per 64-token block and exact
FP16 rerank widths 32/64/128) matched the deployed INT4 top-1 on those states.
That is encouraging but is not a broad quality proof.

Performance is decisively negative. At one row, INT4 plus full-vocabulary
argmax took 1.190 ms median; INT2 shortlist plus rerank took 9.452--9.463 ms.
At 16 rows the corresponding medians were 1.267 ms and 9.576--9.624 ms.
The INT2 representation saves storage (357,580,800 versus 655,564,800 bytes),
but this Xe2 Triton implementation is 7.6--7.9 times slower end to end. It is
therefore rejected without a serving arm.

Evidence: `runs/run-20260915-202642/`. The real activation source is
`runs/capture-20260915-202008/`; the sampler retained V2 static buffers and
deduplicated 256 asynchronous snapshots to 16 distinct draft states.

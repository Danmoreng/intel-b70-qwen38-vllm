# Bounded MTP component profile

Both MTP3/MTP4 arms completed the three selected code cases with128 output
tokens per request. `assessment.json` audits all six event files, four separate
eight-cycle kernel traces, inventories, input identity and raw-result hashes.
These are instrumented diagnostic timelines including host dispatch gaps,
not the serving throughput results.

| C1 code context | MTP | Actual/padded target rows | Pure cycles | Target body ms | Target head ms | Complete draft proposal ms | Whole cycle ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4K | 3 | 4/4 | 36 | 39.32 | 2.19 | 4.07 | 45.84 |
| 4K | 4 | 5/5 | 29 | 40.50 | 2.20 | 5.41 | 48.38 |
| 103K | 3 | 4/4 | 44 | 58.99 | 2.03 | 6.63 | 67.90 |
| 103K | 4 | 5/5 | 42 | 63.38 | 2.04 | 8.82 | 74.49 |

The128-token C1 outputs match between depths in both cases. Longer512-token
serving outputs can diverge later, so this does not close generated-path quality.
Sampler and sampled-token commit spans are small in these captures; other
state-update work outside those call sites remains in the whole-cycle envelope.

Cold nominal C4 mixes prefill/decode and has pure-decode groups with only1/2/3
active requests; do not label those groups as a clean four-request decode test.
The immediately following warm trace does observe four active requests in all
eight captured cycles: target graph bucket16/actual16 for MTP3 and20/actual20
for MTP4. Observed padding is exact in these cases, not an assumed24-row bucket.

The kernel traces expose the full LM-head and surrounding work, but do not
demonstrate individual target-body or fused-draft graph-node visibility. The
MTP3 warm C4 trace has8 full-head DPAS kernel events and no attention-named
kernel. This does **not** prove attention is cheap: target and draft bodies run
inside FULL graphs. A separate bounded eager103K C1 trace is prepared/running
to expose those kernel operations. It will not be mixed into serving throughput.

Both full heads have248320 rows/6bit weights; each also owns a65536-row pruned
draft copy with252315648 logical tensor bytes. Distinguish those logical bytes
from allocator residency and verify ownership/output behavior before any
duplicate-copy removal or full-vocabulary decision.

Remaining: fine attention/draft decomposition, generated-path/margin check,
targeted depth selection/repeats, and full-vocabulary ablation. This is not a
completed performance or release gate.

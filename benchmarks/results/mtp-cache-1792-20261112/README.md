# MTP3 cache1792/128: operator-approved production tradeoff

On 2026-11-12 the operator approved replacing v6's 1600/1600 cache geometry with
1792-token physical pages and 128-token prefix matching, including the complete
dense-alias COW copy fix. Total prefill batch 4096, MTP3, 180 W, 16 configured
slots, max 262144 tokens, LBNHC and 12 GiB RAM ARC remain.

The [full comparison](qualification/full-serving-20261112-r2/comparison.json)
contains 70 valid measured waves / 124 byte-matched request payloads per arm:

| Metric | Previous v6 | 1792/128 | Change |
|---|---:|---:|---:|
| Sum of measured wave wall time | 2423.72 s | 2417.01 s |−0.28% |
| Weighted native decode | 47.68 tok/s | 48.38 tok/s |+1.47% |
| Summed request prefill phases | 1268.19 s | 1311.96 s |+3.45% |
| Card energy | 436.44 kJ | 435.34 kJ |−0.25% |

The warm 16K/64K prefix cases have 9.8–10.4% shorter median request time and
67.8–77.1% shorter median maximum client TTFT. The 1024-output-token serving
matrix disables thinking. Only four measured waves are warm repetitions;
the tiny overall difference is not evidence of a universal engine speedup.
The reference is the same-native v6 run from 2026-11-10, not a same-day ABBA.

There are actual costs: cold short-request TTFT rises about 0.1–0.15 s in some
cells, including 297→444 ms at 512 tokens. Summed overlapping request phases
are not isolated GPU timings or wave wall. Long cold C1 prefill is within
roughly −1.9% to +0.3%. The single 199K decode case drops 6.15%, mostly alongside
lower draft acceptance and 364→387 counted rounds; milliseconds per counted
round rise 0.216%. See the [phase table](qualification/full-serving-20261112-r2/scenario-comparison.csv)
and [review of all 13 automatic flags](qualification/full-serving-20261112-r2/trigger-review.json).

Eight separate genuine tool-agent runs use regular thinking sampling,
thinking budget 8192, natural ends and independent hidden tests. Both arms
solve 4/4 tasks; median paired task-time change is −0.65%, effectively a tie.
This small panel supports functionality, not a broad quality/speed guarantee.

[Finite technical gates](qualification/TECHNICAL-GATES.json) cover controlled
C1/independent C2 state/KV/logit/probability checks, full-page COW copies, the
17920 boundary, genuine RAM restore, abort/reuse/admission and an actually
executed 261120-input + 1024-output = 262144 sequence. Numerical controls never
run in production. This is not two simultaneous 262144-token histories.

## Release packaging and deployment

The large prior qualification measured the v6 image with explicit 1792/128
arguments and the helper installed by RPC. The release child activates that
same byte-identical helper automatically after real KV initialization.
No native kernel is rebuilt. `packaging-identity.json` and
`python-payload-comparison.json` attest the bridge: model/native bytes retained;
only serving YAML, the startup hook, the helper and the attention guard's
exact runner-hash literal change. The source guard remains enabled.

`release-child-gate-r2/` checks the exact release image without developer
endpoints: automatic 65→17 COW view selection, two concurrent continuations
with actual 16256-token partial-page hits, correct remembered facts, parent
recheck and actual GPU→RAM transfers. Production promotion also checks a real
service restart, streaming and API defaults. Receipts are separate from the
historical benchmark data; the full 70-wave matrix is not rerun on packaging.

The first packaging attempt (`release-child-gate/`) was rejected before model
loading by the strict attention source guard. Its expected runner hash was
updated to the reviewed startup-only change; this is not a GPU fault.

## Open stability issue accepted by the operator

The first full 1792 sweep failed after 36/70 waves with a BCS GPU memory fault
at the 128K→C2 transition. Kernel logs and the xe dump are retained locally.
A thin-logging transition replay completed 201 transfers / 21.07 GB, and a
fresh full 70-wave repeat passed without a new fault. Similar fault families
were also observed with original v6; cause and relative frequency remain
unknown. This release does not claim the fault is fixed. The operator
explicitly accepted adoption after reviewing this issue and the cold-latency
costs. The separate corrected COW truncation bug is not a proven cause.

The previous v6 image/policy and RAM runtime are archived under
`config/releases/exl3-v6`; validate its launcher with
`scripts/run-server-exl3-v6-rollback.sh --check-only`. Stop the current service
before loading the rollback on the same GPU/port.

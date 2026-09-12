# Measured results

Measurements were collected on one Intel Arc Pro B70 32 GB with one active
request. Native vLLM prefill/decode counters were used; warmups were excluded.
Results are not cross-hardware claims and should not be read as a model-quality
benchmark.

## Final coding profile: MTP6 versus MTP4

Matched W4A16 runs used an 8,192-token coding prompt, up to 16,384 output
tokens, the improved workload-tuned 40K draft vocabulary, and executable tests.

| Profile | n | Median prefill | Median decode | Mean wall time | Fixed tests |
|---|---:|---:|---:|---:|---:|
| MTP4 control | 2 | 1,927.60 tok/s | 93.22 tok/s | 60.46 s | 26/26 |
| **MTP6 production** | 2 | 1,922.34 tok/s | **103.15 tok/s** | **55.61 s** | 26/26 |

MTP6 improved median decode by 10.7% and reduced mean completed-task time by
8.0%. One MTP6 response also passed all 19 self-generated tests; every output
passed the fixed suite.

## Throughput across context lengths

The long-context W4A16 measurements below predate the final MTP6/40K retune and
used MTP4 with the full INT4 draft head. They remain useful as a reproducible
scaling baseline; they are not mislabeled as final-MTP6 measurements.

| Input tokens | n | Prefill | Decode | Accepted/drafted |
|---:|---:|---:|---:|---:|
| 8,192 | 1 | 1,952.32 tok/s | 89.88 tok/s | 75.2% |
| 65,536 | 1 | 1,226.30 tok/s | 71.65 tok/s | 76.3% |
| 122,880 | 1 | 882.81 tok/s | 59.30 tok/s | 76.6% |

All three were complete coding requests with an output budget of 16,384. The
8K result passed 13/13 fixed and 21/21 generated tests; 65K passed 13/13 and
21/21; 122K passed 13/13 and 19/19.

## Why target W4A8 was rejected

The full coding A/B is the production decision, because prefill-only screens
hide decode regressions.

| Target path | n | Median prefill | Median decode | Mean wall time |
|---|---:|---:|---:|---:|
| **W4A16 production** | 2 | 1,935.86 tok/s | **94.99 tok/s** | **56.59 s** |
| W4A8 prefill experiment | 2 | **2,673.68 tok/s** | 85.16 tok/s | 65.47 s |

W4A8 improved prefill by 38.1%, but decode fell 10.3% and task wall time rose
15.7%. It is therefore not enabled by this repository.

The isolated cold-prefill screen explains where it may still be useful:

| Input tokens | W4A16 | W4A8 | Change |
|---:|---:|---:|---:|
| 8,192 | 1,968.19 tok/s | 2,772.53 tok/s | +40.9% |
| 65,536 | 1,226.30 tok/s | 1,497.22 tok/s | +22.1% |
| 122,880 | 882.81 tok/s | 1,014.40 tok/s | +14.9% |

At 122K, full attention accounted for 62.9% of summed GPU-kernel time, so a
faster GEMM path produces a smaller total-prefill gain as context grows.

## Draft vocabulary selection

Matched 8K/512 screens used three requests per arm:

| Draft head | Prefill | Decode | Accepted/drafted |
|---|---:|---:|---:|
| Full 248,320 rows | 1,968.19 tok/s | 104.48 tok/s | 87.7% |
| 65,536 rows | 1,967.57 tok/s | 111.04 tok/s | 88.3% |
| **40,960 rows** | 1,965.58 tok/s | **112.71 tok/s** | **89.4%** |

The selected list is corpus-dependent and is not included because the measured
version was derived from private coding sessions. The public builder lets every
user create and evaluate their own list without disclosing their corpus.

## Capacity and functional checks

- 215,870 KV-cache tokens available at `gpu-memory-utilization=0.93`.
- 204,800-token configured context passed a boundary request containing vision.
- Prefix-cache test: 17,662 prompt tokens; cold TTFT 10.915 s, warm TTFT
  2.902/2.901 s, 13,312 cached tokens reused on each warm request.
- Vision input, parsed tool call, tool-result continuation, and a real coding
  read/edit/bash smoke test passed.

Run `scripts/benchmark.py` to generate a fresh, privacy-safe context table for
your exact host and final local vocabulary.


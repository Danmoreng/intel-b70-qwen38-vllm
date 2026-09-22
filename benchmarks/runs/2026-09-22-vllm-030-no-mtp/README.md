# Short A/B: vLLM 0.29 vs 0.30 without MTP

Date: 2026-09-22. Completed raw run ID: `v030-nomtp-retry-20260922`.
Wall time including starts, warmups, measurements and production restoration:
24 minutes 42 seconds (23:26:35–23:51:17 CEST). The ignored local
`benchmark-results/v030-nomtp-retry-20260922/` directory contains the raw
prompts, responses, streams, telemetry and server logs. The tracked
[`summary.json`](summary.json) contains aggregates and validation results.

This is a follow-up to the [MTP-enabled A/B](../2026-09-22-vllm-030-short/README.md).
It used the exact same control and candidate images, model revision, 180 W cap,
4K/1K coding prompts, C1/C4 scenarios, greedy sampling, batch-token budget,
and B/A/A/B order. The only serving change was removing
`--speculative-config` from **both** versions. The same 10 prompt payloads
were verified by SHA-256 across all arms and also matched all 10 payloads in
the earlier MTP-enabled run. There were two measured repeats per
scenario per arm after warmup, for 16 measured waves and 40 requests. The
versions were vLLM `0.29.0+xpu` and `0.30.0+xpu`, both with XPU kernels
`0.1.14.1`.

All measured outputs were nonempty and exactly 1024 tokens. Every request
ended with `finish_reason=length`; there were zero draft and accepted tokens,
zero prefix-cache hits, and zero preemptions. Q128 and M04 binaries remained
in both images but neither dispatched for this no-MTP workload. Thus the
numbers compare the native attention path of the two otherwise unchanged
stacks. The server arguments and extra environment were identical across
arms.

## Results

Means over four measured waves per version and scenario:

| Scenario | Metric | vLLM 0.29 | vLLM 0.30 | 0.30 change |
| --- | --- | ---: | ---: | ---: |
| 4K/1K, C1 | Native decode tok/s | 31.834 | 31.837 | **+0.01%** |
| 4K/1K, C1 | Prefill compute tok/s | 1564.0 | 1558.6 | −0.35% |
| 4K/1K, C1 | Batch wall time | 34.761 s | 34.766 s | +0.02% |
| 4K/1K, C4 | Fully overlapped aggregate decode tok/s | 112.042 | 111.956 | **−0.08%** |
| 4K/1K, C4 | Prefill compute tok/s | 468.8 | 463.8 | −1.07% |
| 4K/1K, C4 | Batch wall time | 47.385 s | 47.517 s | +0.28% |

The C1 native decode ranges were 31.811–31.877 tok/s on 0.29 and
31.791–31.900 tok/s on 0.30; their means differ by just 0.003 tok/s. The
earlier MTP-enabled screen showed +1.46% C1 native decode and −11.03% C4
fully overlapped aggregate decode for 0.30. Neither effect persists here.
This supports investigating vLLM 0.30's interaction with MTP, but does not
identify the cause or prove that MTP acceptance alone explains it. Exact
output hashes again varied between repeated control arms (1 of 10 matched);
that is not an isolated 0.30 correctness finding.

**Decision:** No evidence of a meaningful vLLM 0.30 performance gain without
MTP for these scenarios. Do not promote it on this result. The production
0.29 image and 180 W cap were restored, and `/health` returned 200. The
initial no-MTP attempt (`v030-nomtp-20260922`) ended after its first arm
because the runner incorrectly required Q128 dispatch. It restored
production; none of its data is included in the completed A/B aggregate.

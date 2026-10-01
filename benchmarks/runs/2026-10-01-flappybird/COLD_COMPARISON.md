# Matched coding histories: cold throughput and preemption diagnostics

The original strict control required zero preemptions. EXL3 failed that criterion at 139K, also on a fresh worker. Its exact-count, zero-cache observations are reported explicitly as preempted diagnostics; no failed measurement is silently relabelled as a clean control. GPTQ points have zero preemptions. All six inputs, native counter differences, output counts and model identities were audited.

| Actual input | GPTQ cold prefill | EXL3 cold prefill | GPTQ decode | EXL3 decode | Preemptions GPTQ / EXL3 | Classification |
|---|---:|---:|---:|---:|---:|---|
| 102,752 | 1,373.4 | 1,397.2 | 64.5 | 46.1 | 0 / 0 | Clean control |
| 139,193 | 1,196.5 | 1,176.1 | 37.5 | 33.7 | 0 / 1 | Preempted diagnostic |
| 187,695 | 1,030.6 | 1,002.3 | 29.8 | 24.4 | 0 / 1 | Preempted diagnostic |

Rates are tokens per native phase second. Prefix-cached tokens are zero; all input tokens are computed, and exactly 1,024 outputs are generated with ignored EOS. Decode counts the 1,023 tokens after the first. Model ID is the only profile-specific payload field. Sampling and seeds match the original archived request; generated continuations may differ. Outputs are truncated and tool calls are not executed.

The EXL3 139K and 188K requests use separate fresh workers after excluded warmup. The GPTQ three-point series and EXL3 103K point come from the initial fresh-worker control. Cold token accounting is verified for every observation. These probes run after game reviews, without simultaneous browser testing. They compare complete pinned serving recipes, not isolated quantization or kernels. The primary adaptive v6 sessions had zero preemptions throughout.

See cold-comparison.json for native before/after counters, worker identities, exact hashes and the rejected zero-preemption attempt. The compressed original prompts make the common input reviewable.

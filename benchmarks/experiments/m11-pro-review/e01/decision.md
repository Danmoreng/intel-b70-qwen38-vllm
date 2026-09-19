# E01 decision: do not promote

Five paired 8K/16K serving blocks completed on 2026-09-19. **No reliable serving improvement**. The predeclared short-serving gate failed; E01 remains an isolated opt-in experiment. No production image, launcher or service configuration was promoted.

| Context | Cache | Prefill latency reduction | Decode latency reduction | Decode paired bootstrap 95% interval |
|---|---|---:|---:|---:|
| 8192 | cold | +0.02% | +0.94% | [-0.53%, +3.09%] |
| 8192 | warm | +0.01% | +0.29% | [-0.91%, +1.68%] |
| 16384 | cold | +0.06% | -0.35% | [-2.98%, +2.21%] |
| 16384 | warm | -0.07% | +0.30% | [-0.44%, +1.06%] |

Positive values mean lower latency. These are means of paired latency changes, not ratios of separately averaged token rates. All decode intervals cross zero. None of the affected-phase cells establishes the practical 2% improvement gate.

The comparison used the same image with only `B70_FUSED_QK_ROPE_GATE=0/1` changed, fixed 180 W, 4096 prefill budget, MTP4, Q128/M04, FP8 KV, FP16 target head and 200704 model limit. Exactly 256 generated tokens per request; prompt hashes, computed tokens and cache hits match within each pair. Cold means 8192/16384 computed tokens; warm means 4864/6400 computed tokens after 3328/9984 cache hits. No preemptions. Native phase counters were read only after request accounting completed.

Quality gates passed: 180 synthetic numerical cases, 26 captured model-activation cases including vision and MTP, exact gate/V output, graph replay, 3/3 coding assertions in both serving arms, vision/tool continuation and prefix-state checks. Mean teacher-forced NLL change was -0.000109 nats/token across the three small samples; these are limited screens, not proof of unchanged quality for all tasks.

The isolated QK preparation region was faster, but its reference was a set of installed calls captured in a graph. The complete serving reference is already TorchInductor-compiled. The trace confirms 51 candidate fused-kernel calls, so the absent serving gain is not an inactive flag.

**32K/64K, full-context generation and coding long runs were not run for E01**, because the preliminary serving gate did not pass. Startup retained the configured 200704 context, which alone is not full-context qualification. The earlier high-context operator tests are also not long-context serving benchmarks.

Raw evidence stays under `runs/serving-20260919-200901`; compact data is in `results.json`. Earlier stopped harness attempts are retained separately and excluded from these five pairs. E02 remains skipped by user instruction.

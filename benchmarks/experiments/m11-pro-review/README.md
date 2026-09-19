# Pro review implementation — 2026-09-19

Sequential experiment programme based on the attached review. The review is
source analysis, not evidence of GPU gains. Production reference is the promoted
Runtime 26.35 / IGC 2.41.5 image with Q128, M04, MTP4 at 180 W and 200704 tokens.

1. E01: opt-in QK/RoPE/raw-gate fusion; first operator numerical/graph screen,
   then captured-activation and serving gates if it passes.
2. E02: **SKIPPED by user instruction (2026-09-19)**. Prefill budgets were
   already tested repeatedly without benefit. Keep the production budget 4096;
   do not rerun the review's proposed budget sweep.
3. E03: M04 final-store layout, only after relevant profile evidence.
4. E04: constant Gemma norm weights, only if remaining runtime work is measured.
5. E05: GDN scratch initialization, only with native provenance and write coverage.

Keep/drop decisions must be recorded individually. No MTP retuning, context
reduction, target-head quantization or sampler change. Production promotion
requires correctness, clear serving benefit and full-context/vision/tool/prefix
qualification. Synthetic operator inputs cannot stand in for model quality.

Each GPU experiment acquires the existing exclusive lock, checks idle production,
and installs both a finally recovery and a systemd ExecStopPost recovery.

## Completed decisions

- [E01](e01/decision.md): implemented and tested; 180 synthetic and 26 captured
  activation cases plus small serving quality checks passed. Five paired
  8K/16K blocks found no reliable serving gain; not promoted. No long-context
  serving or coding long run was started after the short gate failed.
- [E02](e02/decision.md): skipped by the user's explicit instruction.
- [E03](e03/decision.md): measured output-layout cost on the production binary
  at 8K/16K/64K/196K and M=2/3/4/5. Even the deliberately invalid output-copy
  elision remained below 2% of the M04 region; no native implementation justified.
- [E04](e04/decision.md): transformation already fused into observed compiled
  norm kernels; no measured remaining bottleneck to justify a weight cache.
- [E05](e05/decision.md): small observed initialization share and unresolved
  source-to-wheel equivalence; prerequisite for a native scratch change not met.

No experiment in this review series was promoted. The Runtime 26.35 / IGC 2.41.5
production image, Q128/M04, MTP4, 4096 prefill budget and 200704 context remain the
reference. Compact results are retained beside each decision; raw traces,
activations, compiler caches and request streams stay in ignored run directories.

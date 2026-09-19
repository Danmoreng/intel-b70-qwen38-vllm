# Performance experiments

The immutable control is the promoted Q128, MTP4, full-vocabulary, 180 W
configuration. Experiments must not modify its image or production service.

Each run records an attributable manifest, exact request hashes, engine logs,
metrics, memory/power samples, a machine-readable result and a decision note.
Large raw traces, SSE streams and generated requests stay local and are ignored;
compact manifests and summaries may be reviewed and committed separately.

The M01 observability run is deliberately diagnostic rather than a throughput
promotion run. It consists of:

1. the frozen 200448-input/256-output long-context replay with 250 ms memory,
   power and scheduler sampling; and
2. an 8192-input/256-output full decode trace whose profiler starts before XPU
   graph capture and stops after the real request.

MTP depth remains fixed at four in both phases.

Current decisions:

- M03 custom GPTQ small-M linears: rejected; oneDNN remains substantially faster.
- M04 shared-KV verification attention: promoted after +5.97% decode at 65K
  and +3.09% at the 200704-token boundary.
- M05 INT2 draft head: rejected; good small-sample recall but 7.6--7.9x
  slower than the deployed INT4 head on Xe2.
- Target-head INT4: serving repaired and screened on September 19, followed
  by the paired coding run. The control completed both tasks (9/9 acceptance);
  the candidate exhausted its output-token budget during the follow-up (8/9).
  Not promoted; production retains the FP16 target head. See
  [the coding results](m06-target-head-int4/CODING-RESULTS-2026-09-19.md) and
  [the short serving report](m06-target-head-int4/SERVING-2026-09-19.md).
## M10 GDN input projection

[M10 mixed INT4/FP16 projection](m10-gdn-projection/README.md) was implemented
on the promoted Intel Runtime 26.35 image. Numerical checks passed 15/15;
captured-graph device latency regressed to 1.9–2.8x the existing pair. Rejected
before 8K/16K serving tests or any coding long run. Production was restored.

## M11 Pro review follow-up

[Sequential review implementation](m11-pro-review/README.md): E01 passed
operator and small quality checks but found no reliable serving gain in five
paired 8K/16K blocks; not promoted. E02 was skipped at the user's request.
E03's measured output-layout cost was small even within the M04 region.
E04/E05 did not meet their profile/provenance prerequisites. Production remains
on the promoted Runtime 26.35 / IGC 2.41.5 image with Q128/M04 and MTP4.

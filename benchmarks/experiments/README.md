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
- Target-head INT4: promising offline screen only; quality-changing and not
  eligible for serving until broader NLL/task qualification passes.

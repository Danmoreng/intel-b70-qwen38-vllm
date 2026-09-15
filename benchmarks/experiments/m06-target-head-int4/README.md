# M06 INT4 target-head offline gate

This arm compares the unchanged FP16 target head with INT4/G128 on captured
real target hidden states. It measures operator speed, top-1 agreement,
top-20 recall, logit error and KL divergence. Unlike draft-head quantization,
this changes the target distribution; a positive speed screen is not enough
for serving promotion. NLL and representative task qualification are mandatory.

## Current result: offline-positive, not qualified

The clean V2-runner capture yielded 36 distinct real target states. INT4
matched FP16 top-1 for all 36, retained 96.11% of the FP16 top-20 set, and had
mean KL(FP16 || INT4) 0.00463 (maximum 0.02716). At 36 rows, full projection
plus argmax improved from 4.868 to 1.468 ms (+231.7%). This justifies a broader
quality arm, not production use. The earlier static-buffer captures are invalid
for quality conclusions and are intentionally not summarized as evidence.

# QueueKit native-attention control

A single diagnostic control follows the final release measurement campaign.
It repeats the same two frozen QueueKit v2 tasks, seed schedule, sampler and
client limits on the same immutable EXL3 candidate. Its only runtime override
is `EXL3_GUARDED_ATTN=0`, disabling the guarded oneDNN/M04 attention dispatcher.
This is an experimental control, not the frozen production profile.

The optimized result remains 1/2 tasks and 7/8 acceptance cases; its generated
code is never repaired or replaced. The control helps interpret that outcome
without establishing a general model-quality ranking from one pair.

`recorded-controller.py` is the exact local controller snapshot, including its
original paths and fresh-output guard. The worker receipt records the complete
Docker command/environment. To repeat, choose a fresh output path and the
matching completed final campaign; preserve the fixture and client settings.
The final release report records both outcomes and their limits.

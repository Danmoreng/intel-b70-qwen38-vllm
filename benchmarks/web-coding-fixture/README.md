# Reproducible long web coding task

Use **[v7](v7/README.md)** for a new run. It is a small deterministic WebGL2
Flappy Bird game with controls, responsive layout, settings and local highscores.
There is no level editor, gameplay replay system, backend or runtime library.
The starter, stage prompts, sampling and harness hashes are frozen.

The [overnight paired results](../runs/2026-10-01-flappybird/README.md) used
**v6**, reaching 153K GPTQ / 106K EXL3 context. A storage-error harness assumption
was corrected only after timing, with raw and contract-reviewed scores both
retained. V7 keeps the same starter and instructions, corrects that assumption,
and is ready for future measurements; it has not been timed in this campaign.

[Calibration history](../web-coding-calibration-20261001.md) documents every
scope change. V5 was a correct but short 83K game, so v6 added settings and
highscores as authorized. Earlier fixtures and the [v4 stress comparison](../runs/2026-10-01-flappybird-stress-v4/README.md)
remain available as separate evidence, not pooled into the headline pair.
Historical fixtures pin historical runner revisions and are not substitutes for
v7 with the current runner. Context coverage is observed, never padded or
forced; an equal budget stop is an incomplete task.

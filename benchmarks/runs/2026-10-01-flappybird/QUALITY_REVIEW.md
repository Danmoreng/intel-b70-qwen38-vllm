# Final game output review

Both unmodified v6 outputs draw visible bird and pipe bodies, leave the pipe
gaps open, produce no WebGL error, and leave the simulation state unchanged
while rendering. The same gap-following controller drives each actual mounted
app through **1,500 ticks and 12 points** without a gameover or page error.
The browser uses CPU SwiftShader and blocks external runtime requests.
These are supplemental probes, outside the frozen score and native timings.

| Observation | GPTQ production v2 | EXL3 4.00 bpw |
|---|---|---|
| Frozen final functional checks | 51/54 | 54/54 |
| Checks after correcting the storage-error probe | 52/54 | 54/54 |
| External persisted-schema / real storage error probes | 3/3 | 3/3 |
| Visible bird, top/bottom pipes, open gap | Passed | Passed |
| Actual mounted game: 1,500 ticks / 12 points | Passed | Passed |
| Task explicitly completed within shared budget | No | Yes |

## The storage test correction

The public score-store API requires storage failures to propagate as errors.
It does not require the factory to read storage eagerly. The frozen v3 Node
probe incorrectly expected `getItem` to throw during construction. GPTQ loads
lazily and correctly propagates the error when `list()` accesses storage.
The corrected v4 probe accepts errors during construction **or actual access**
and likewise checks write failure during construction or `record()`.
Both projects pass; the raw frozen scores, timed feedback and generated code
remain unchanged. JSON contract-review artifacts include the original score,
project digest, corrected check hash and unchanged browser check hash.

This distinction matters to task time: GPTQ received an overstrict *visible*
test failure during the timed session. Its 40-minute stop therefore cannot be
attributed purely to coding ability or quantization. The corrected **fixture
v7** is supplied for future repetitions; it has not been timed in this report.
The actual paired measurements used v6, with the same frozen harness on both
sides. No post-run correction removes past model requests or timing.

## Remaining GPTQ checks

Two frozen browser cases fail at the same missing selector:
`input[data-testid="player-name"]`. The actual labelled name input exists,
is editable, and is wired into the score entry; it has `id="player-name"`
but lacks the publicly requested `data-testid`. This is an integration-contract
omission, not evidence that the name field or all persistence is absent.
Because those browser cases stop at that selector, they do not fully establish
GPTQ's keyboard-isolation and save/remount behavior. Do not turn this report
into a claim that those behaviors passed. EXL3 passes both complete cases.
No generated source has been repaired to improve a grade.

Both applications provide procedural WebGL2 graphics, controls, settings,
highscores and a user guide. Screenshots and original sources accompany the
metrics. This single adaptive pair supports a task-specific outcome, not a
broad code-quality or INT4-versus-EXL3 ranking.

## Timing qualification

A roughly 4.6-second CPU-only GPTQ supplemental review overlapped EXL3 requests
4–5, while the B70 remained exclusive to EXL3 inference. The exact spans are in
[supplemental-review-timing.json](supplemental-review-timing.json). Native token
accounting remains valid, but host-load influence cannot be ruled out. The
separate matched-prompt throughput control begins only after all game reviews
and browser processes have finished.

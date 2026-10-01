# Independent review of the initial WebGL stress attempt (v4)

Neither final project renders a usable game. The uniform functional-case score
is not severity-weighted: a single renderer failure can hide every gameplay
object despite passing simulation and UI checks.

- **GPTQ: 35/45 cases.** The timed session exhausted its 200,704-token window
  after 98 requests; release was never reached. Its final `app.js` contains
  assignments such as `canvas.dataset-testid = ...`, which are JavaScript
  syntax errors. Node and the real browser reject the module. In the renderer,
  coordinate helpers divide by width/height arguments that their callers omit,
  producing non-finite positions; the draw count also divides two-float XY
  vertices by three. The supplemental scene has no visible bird or pipe pixels.
  Patching the global canvas `getContext`/`readPixels` path did not fix geometry.
- **EXL3: 44/45 cases.** The agent declared five of six stages complete; graphics
  reached its request cap. UI controls and the actual simulation work: the
  supplemental controller advances 1,500 ticks and scores 12 points. However,
  pixel coordinates are sent directly to clip-space `gl_Position` without the
  required conversion, clipping gameplay objects. The frozen renderer case
  sees two colors, and the separate canonical scene confirms missing bird and
  pipe pixels. The high score does not establish a playable rendered game.

The output trees were graded read-only and remain unmodified. Saved screenshots
and `*-visual-review.json` record the observations. These review probes were
prepared before the completed outputs were inspected and do not change the
45-case score or native/session timings.

A second frozen variant (v5) gives both profiles the same textbook coordinate
conversion and vertex-stride reminder. It keeps the original starter, other
five stage instructions, game scope, sampler and acceptance harness. Its fresh
results are reported separately; these failures are not silently discarded or
combined with the second sample. Selection of a usable benchmark is exploratory;
neither pair establishes a general quantization-quality ranking.

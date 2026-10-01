# FlappyLab user guide

A small, offline, deterministic WebGL2 Flappy Bird lab. No installs, no build step, no external services.

## Running

Serve the project statically, for example:

```
python3 -m http.server 8765
```

then open http://localhost:8765 in a modern browser (WebGL2 required).

## Controls

- **Flap**: `Space` or `ArrowUp`, or tap/click the canvas. `Space` also starts the game from the ready screen.
- **Pause / resume**: `P` (or the Pause button, which toggles to Resume).
- **Restart**: `R` (or the Restart button).
- Game hotkeys are ignored while a text field is focused, and held keys do not re-trigger.
- The game auto-pauses when the browser tab becomes hidden.
- The layout is responsive; the canvas rescales to small screens and high-DPR displays (DPR capped at 4), with visible keyboard focus rings and `prefers-reduced-motion` support.

## Settings (seed, gravity, gap)

The settings panel has three labelled number inputs:

- **Seed**: any 32-bit unsigned integer. Same seed ⇒ identical pipe sequence, so runs are reproducible and replayable via snapshots.
- **Gravity**: 0.01–2 (per-tick).
- **Gap height**: integer 60–(height − ground − 40).

Press **Apply settings** to take effect: the values are validated together against the full config rules, and on success *only* seed/gravity/gapHeight change, the game restarts to ready, and your saved high scores are kept. Invalid input shows an error message, changes nothing, and the current game continues exactly as it was.

## Player name and high scores

- Set your **Player name** (trimmed, up to 24 characters; defaults to "Player").
- Every time a running game ends, exactly one score entry (`name`, `score`, `frames`, `seed`) is saved to the local browser storage under the key `flappybird.scores.v1` — it survives reloads and re-mounting the page.
- The ten best results are kept, sorted by score (fewer frames break ties), and rendered as name/score rows.
- **Clear scores** empties the list locally; nothing is ever sent anywhere.

There is deliberately no level editor, replay tooling, or backend.

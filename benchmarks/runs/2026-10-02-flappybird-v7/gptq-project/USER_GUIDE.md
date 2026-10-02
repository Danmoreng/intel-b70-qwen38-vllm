# FlappyLab User Guide

A deterministic, offline WebGL2 Flappy Bird laboratory. No accounts, no network,
no third-party libraries.

## Run it

Any static file server works; for example from this folder:

```
python3 -m http.server 8765
```

then open http://localhost:8765.

## Playing

- **Space** / **Arrow Up** / tap or click the canvas: start the game or flap.
- **P** (or the **Pause** button): pause; press again to resume.
- **R** (or the **Restart** button): reset the run with the same configuration.
- The score and status are shown above the buttons; screen readers receive the
  same updates through a polite live status region.
- The canvas resizes with the window (device-pixel-ratio aware, capped at 4x),
  and the game pauses automatically when the tab is hidden.

## Settings

The settings panel has labelled number fields for **Seed**, **Gravity** and
**Gap**, each pre-filled with the current game configuration, plus an
**Apply settings** button and an error line.

- Applying valid values validates the full configuration, updates only those
  three settings, and starts a fresh ready-state game.
- Invalid values (empty, out of range, or breaking a joint rule such as gap
  size versus play-field height) are rejected with a visible error; the running
  configuration and current game state are left untouched.
- The game's keyboard shortcuts (Space, P, R) are ignored while a settings or
  name field has focus.

## Scores

- The **Player name** field (default "Player", up to 24 characters, trimmed on
  save) names the score for the current run.
- When a running game first ends, exactly one score entry
  (`name, score, frames, seed`) is saved; further frames or restarts after the
  crash never duplicate it.
- The best ten scores are listed as rows ("name - score"), sorted by score
  then fewest frames, and persisted in your browser under the key
  `flappybird.scores.v1`, so they survive reloads.
- **Clear scores** empties the list, also persisting the reset.

## Determinism

The simulation steps at a fixed 60 Hz and draws pipe gaps from a seeded
xorshift RNG; it never reads the wall clock or `Math.random`. The same seed,
settings and flap timings always reproduce the identical run.

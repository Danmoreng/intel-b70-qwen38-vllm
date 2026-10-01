# FlappyLab User Guide

A small, deterministic, offline Flappy Bird built with vanilla WebGL2. No build step, no dependencies.

## Run it

Serve the project directory with any static HTTP server, for example:

```sh
python3 -m http.server 8765
```

Then open `http://localhost:8765` in a modern browser (WebGL2 required).

## Controls

- **Flap**: `Space` or `ArrowUp`, or tap/click the canvas. Flap also starts a ready game.
- **Pause / resume**: `KeyP` (toggles) or the Pause/Resume button.
- **Restart**: `KeyR` or the Restart button.
- Buttons can also be used with the keyboard (Tab + Enter).
- The game auto-pauses when the browser tab is hidden.
- Keyboard shortcuts are ignored while an input or textarea is focused.

## Deterministic seed

The simulation is fully deterministic: pipe gaps come from a seeded xorshift32 RNG (default seed `12345`), so the same seed and input sequence always produce the same game. Physics uses fixed 60 Hz steps, independent of frame rate.

## Game

Pass through the pipe gaps to score one point per pipe. Touching a pipe or the ground/ceiling ends the game. The score and phase (ready/running/paused/game over) are shown above the canvas, and state changes are announced politely to screen readers.

## Settings

Below the game you can change:

- **Seed**: the RNG seed for pipe generation (0–4294967295). Same seed = same pipe layout.
- **Gravity**: fall acceleration (0.01–2).
- **Gap height**: the pipe opening height (integer, limited by the playfield).

Press **Apply settings** (or the button) to validate and apply. Valid values restart the game into the ready state with the new seed/gravity/gap; invalid values leave the current game config and state untouched and show an error message next to the button. Your high score list is never affected by applying settings.

## High scores

- Enter your **Player name** (trimmed, up to 24 characters) before you play.
- Every time a running game ends, your result (name, score, frames, seed) is saved once to the local browser (`localStorage` key `flappybird.scores.v1`). Playing again after a game over saves a new entry; the game never double-saves the same run.
- The top ten results (best score, then fewest frames) are listed under **High scores** and persist across page reloads and re-mounts.
- **Clear scores** removes all saved results.

Game keyboard shortcuts (Space/ArrowUp/KeyP/KeyR) are ignored while a settings or name field is focused.

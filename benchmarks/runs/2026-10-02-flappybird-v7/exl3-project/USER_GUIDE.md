# FlappyLab User Guide

A small, deterministic WebGL2 Flappy Bird laboratory. No backend, no third-party libraries.

## Run it

Serve the project root with any static HTTP server (modules do not load from `file://`):

```sh
python3 -m http.server 8765
# then open http://localhost:8765
```

## Controls

- **Flap / start**: `Space`, `ArrowUp`, or tap/click the canvas (a flap from the ready screen starts the game).
- **Pause / resume**: `KeyP` (the **P** key) or the **Pause** button.
- **Restart**: `KeyR` or the **Restart** button (resets to the ready screen with the same configuration).
- **Start**: the **Start** button starts (or resumes) the game from the ready screen.

The keyboard is ignored while an input or textarea field has focus, and the game
pauses automatically when the browser tab is hidden.

## Deterministic seed

The game's pipe layout and all randomness come from a seeded xorshift32 RNG
(default `seed = 12345`). The same seed always produces the same pipe sequence,
so runs are reproducible for comparison and testing.

## Settings

The **Settings** panel has three labelled number inputs — **Seed**,
**Gravity** and **Gap height** — pre-filled with the current game values, and
an **Apply** button. Applying valid values validates the complete
configuration, updates only `seed`, `gravity` and `gapHeight`, and restarts
the game to the ready screen with the new settings. Invalid values are
rejected with a visible error message and leave the current configuration and
game state completely untouched. The game's keyboard shortcuts do nothing
while a text field is being edited.

## High scores

Enter your name in the **Player name** field (default `Player`; names are
trimmed, up to 24 characters). When a running game ends, your result is saved
to the browser's `localStorage` under the key `flappybird.scores.v1` — exactly
once per game, and the list keeps the best ten scores (highest score first,
fewest frames first on ties). The list survives page reloads and re-mounts.
Use the **Clear scores** button to empty it.

## Accessibility

Score and status are shown in dedicated elements, and a `role="status"`
`aria-live="polite"` region announces phase and score changes. Settings
problems appear in an alert region. Buttons and fields are keyboard focusable
with a visible focus ring, and the layout adapts to small/mobile screens and
reduced-motion preferences.

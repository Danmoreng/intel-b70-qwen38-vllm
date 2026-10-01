# FlappyLab User Guide

A small offline WebGL2 Flappy Bird laboratory. No accounts, no network.

## Run it

Serve the project with any static file server, then open the page:

```
python3 -m http.server 8765
# open http://localhost:8765
```

## Controls

- Flap: `Space`, `ArrowUp`, or tap/click the canvas.
- Pause/resume: `KeyP` (or the Pause button).
- Restart: `KeyR` (or the Restart button).
- Start: the Start button, or press a flap key on the ready screen.

The game pauses automatically when the tab is hidden.

## Deterministic seed

Pipe placement is driven by a seeded xorshift32 RNG using the config
`seed` (default `12345`). The same seed always produces the same pipe
sequence, so every run with the default config starts identically.

## Accessibility

- Visible focus outlines for keyboard navigation.
- Score and status are announced through a live region.
- Honors `prefers-reduced-motion` (no CSS motion) and high-contrast
  preferences in the palette.

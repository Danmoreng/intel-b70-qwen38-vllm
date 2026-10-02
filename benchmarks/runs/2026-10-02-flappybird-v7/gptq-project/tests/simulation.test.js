import { test } from 'node:test';
import assert from 'node:assert/strict';
import {createGame, tick, flap, pause, resume, restart, cloneState, restoreState} from '../src/simulation.js';
import {validateConfig} from '../src/config.js';
import {createRng} from '../src/random.js';

const CFG = validateConfig({});
const H = 720, G = 60, GR = 0.35, R = 12;

test('createGame shape, determinism, and detached no-ops', () => {
  const a = createGame({});
  const b = createGame({});
  assert.notEqual(a, b);
  assert.deepEqual(a, b, 'same config gives identical state');
  assert.equal(a.version, 1);
  assert.equal(a.phase, 'ready');
  assert.equal(a.frame, 0);
  assert.equal(a.score, 0);
  assert.equal(a.rngState, 12345);
  assert.deepEqual(a.bird, {x: 120, y: (H - G) / 2, vy: 0});
  assert.deepEqual(a.pipes, []);

  const frozen = structuredClone(a);
  Object.freeze(frozen.config);
  const t = tick(frozen);
  assert.notEqual(t, frozen);
  assert.notEqual(t.pipes, frozen.pipes);
  assert.notEqual(t.bird, frozen.bird);
  assert.equal(t.frame, 0, 'tick on ready is a no-op copy');
  assert.equal(t.phase, 'ready');

  const f = flap(frozen);
  assert.equal(f.phase, 'running');
  assert.equal(f.bird.vy, -6);
  assert.equal(f.bird.y, (H - G) / 2, 'flap does not move the bird');
  assert.equal(frozen.phase, 'ready', 'input state never mutated');
  assert.equal(flap(f).phase, 'running');

  const c = cloneState(f);
  assert.notEqual(c, f);
  assert.notEqual(c.bird, f.bird);
  assert.notEqual(c.pipes, f.pipes);
  assert.deepEqual(c, f);
  assert.deepEqual(restart(f), a, 'restart resets to a fresh ready state');
});

test('fixed-step physics, seeded spawn, pause/resume, ground death', () => {
  let s = flap(createGame({}));
  assert.equal(s.bird.vy, -6, 'flap sets vy; gravity applies on the next tick');
  assert.equal(s.bird.y, 330);

  // Keep the bird aloft until frame 100, where the first pipe spawns.
  for (let i = 0; i < 100; i += 1) {
    s = tick(s, {flap: s.bird.y > 400});
  }
  const firstNext = createRng(12345).next();
  const expectedGapY = 20 + Math.floor(firstNext * (H - G - 180 - 40 + 1));
  assert.equal(s.frame, 100);
  assert.equal(s.pipes.length, 1);
  assert.deepEqual(s.pipes[0], {id: 100, x: 480, gapY: expectedGapY, passed: false});
  assert.equal(s.rngState, createRng(12345).uint32(), 'rngState saved after spawn draw');

  s = pause(s);
  assert.equal(s.phase, 'paused');
  const frozenStep = tick(s, {flap: true});
  assert.equal(frozenStep.phase, 'paused');
  assert.equal(frozenStep.frame, 100, 'tick while paused is a no-op');
  assert.equal(flap(s).phase, 'paused', 'flap while paused is a no-op');
  s = resume(s);
  assert.equal(s.phase, 'running');
  assert.equal(pause(createGame({})).phase, 'ready', 'pause only from running');
  assert.equal(resume(createGame({})).phase, 'ready', 'resume only from paused');

  // Now fall to the ground without flapping.
  let live = flap(createGame({}));
  let n = 0;
  while (live.phase === 'running' && n < 200) {
    live = tick(live);
    n += 1;
  }
  assert.equal(n, 63, 'ground collision at frame 63');
  assert.equal(live.phase, 'gameover');
  assert.equal(live.score, 0, 'no points on the colliding frame');
  assert.equal(tick(live, {flap: true}).phase, 'gameover');
  assert.equal(flap(live).phase, 'gameover', 'flap on gameover is a no-op');
  assert.equal(pause(live).phase, 'gameover');
  assert.equal(resume(live).phase, 'gameover');
});

test('restoreState validates snapshots; scoring and pipe collision', () => {
  const base = {
    version: 1,
    config: validateConfig({}),
    frame: 0,
    score: 0,
    rngState: 1,
    bird: {x: 120, y: 330, vy: 0},
  };
  const invalid = [
    {...base, version: 2},
    {...base, phase: 'flying'},
    {...base, frame: -1},
    {...base, score: 1.5},
    {...base, rngState: 0},
    {...base, rngState: 0x100000000},
    {...base, bird: {x: 120, y: NaN, vy: 0}},
    {...base, pipes: [{id: 1, x: 10, gapY: 100, passed: false},
      {id: 1, x: 20, gapY: 100, passed: false}]},
    {...base, pipes: [{id: 1, x: NaN, gapY: 100, passed: false}]},
    {...base, pipes: [{id: 1, x: 10, gapY: 100, passed: 'yes'}]},
    null,
  ];
  for (const bad of invalid) {
    assert.throws(() => restoreState(bad), Error);
  }

  // Pipe behind the bird: scored on the next tick; snapshot never mutated.
  const snap = {
    ...base,
    phase: 'running',
    pipes: [{id: 5, x: 40, gapY: 240, passed: false}],
  };
  const s1 = tick(restoreState(snap));
  assert.notEqual(s1.pipes, snap.pipes, 'pipes are detached copies');
  assert.equal(s1.pipes[0].x, 38, 'existing pipes move on the first tick');
  assert.equal(s1.score, 1);
  assert.equal(s1.pipes[0].passed, true);
  assert.equal(snap.pipes[0].passed, false, 'source snapshot not mutated');

  // Pipe overlapping the bird but above the gap: collision when the
  // falling bird's bottom edge reaches gapY+gapHeight.
  const snap2 = {
    ...base,
    phase: 'running',
    pipes: [{id: 9, x: 110, gapY: 300, passed: false}],
  };
  let s2 = restoreState(snap2);
  let k = 0;
  while (s2.phase === 'running' && k < 200) {
    s2 = tick(s2);
    k += 1;
  }
  assert.equal(k, 28, 'pipe collision when bird crosses the gap bottom');
  assert.equal(s2.phase, 'gameover');
  assert.equal(s2.score, 0, 'no points awarded on the colliding frame');
  assert.equal(s2.pipes[0].passed, false);
});

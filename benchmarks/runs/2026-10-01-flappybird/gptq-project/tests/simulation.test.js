import test from 'node:test';
import assert from 'node:assert/strict';
import {createGame, tick, flap, pause, resume, restart, cloneState, restoreState} from '../src/simulation.js';

test('createGame shape; pure transitions; flap/pause/resume/restart lifecycle', () => {
  const g = createGame({seed: 9});
  assert.equal(g.version, 1);
  assert.equal(g.phase, 'ready');
  assert.deepEqual(g.bird, {x: 120, y: (720 - 60) / 2, vy: 0});
  assert.equal(g.score, 0);
  assert.equal(g.frame, 0);
  assert.deepEqual(g.pipes, []);
  assert.equal(g.rngState, 9);
  // tick while ready: detached, unchanged.
  const frozen = JSON.stringify(g);
  const same = tick(g);
  assert.notEqual(same, g);
  assert.notEqual(same.bird, g.bird);
  assert.equal(JSON.stringify(same), frozen);
  // flap starts the run and sets flap velocity.
  const run = flap(g);
  assert.equal(run.phase, 'running');
  assert.equal(run.bird.vy, -6);
  assert.equal(g.phase, 'ready', 'input state untouched');
  // pause/resume gate on phase; wrong-phase calls are detached no-ops.
  assert.equal(pause(g).phase, 'ready');
  const paused = pause(run);
  assert.equal(paused.phase, 'paused');
  assert.equal(resume(run).phase, 'running');
  assert.equal(resume(paused).phase, 'running');
  // restart resets with the same config.
  const back = restart(paused);
  assert.equal(back.phase, 'ready');
  assert.equal(back.frame, 0);
  assert.equal(back.score, 0);
  assert.deepEqual(back.config, run.config);
  assert.deepEqual(back.bird, g.bird);
  assert.deepEqual(cloneState(g), g);
  assert.notEqual(cloneState(g), g);
  assert.throws(() => createGame({gravity: 9}));
});

test('deterministic ticks, spawn, scoring, offscreen removal, gameover, no award on colliding frame', () => {
  const cfg = {width: 400, height: 500, birdX: 120, birdRadius: 12, gravity: 0.35, flapVelocity: -6,
    pipeSpeed: 2, pipeWidth: 64, gapHeight: 180, spawnEvery: 10, groundHeight: 60, seed: 7};
  let a = tick(createGame(cfg), {flap: true});
  let b = tick(createGame(cfg), {flap: true});
  assert.deepEqual(a, b, 'same seed, same input -> identical states');
  for (let i = 0; i < 9; i++) {
    a = tick(a, {});
    b = tick(b, {});
    assert.deepEqual(a, b);
  }
  a = tick(flap(createGame(cfg)), {flap: true});
  b = tick(flap(createGame(cfg)), {flap: true});
  assert.deepEqual(a, b);
  for (let i = 0; i < 9; i++) {
    a = tick(a, {});
    b = tick(b, {});
    assert.deepEqual(a, b);
  }
  assert.equal(a.score, 0);
  assert.equal(b.frame, 10);
  assert.equal(b.pipes.length, 1);
  assert.equal(b.pipes[0].id, 10);
  assert.equal(b.pipes[0].x, 400, 'pipes move before a new one spawns at x=width');
  assert.equal(b.rngState, 1892583);
  assert.equal(b.bird.y, 220 - 60 + 0.35 * 55, 'gravity accumulated over ten ticks');
  // A held flap input is not treated as a flap.
  assert.equal(tick(b, {flap: 'yes'}).bird.vy, b.bird.vy + cfg.gravity);
  // Passing: right edge strictly left of the bird awards once; at the boundary it does not.
  const scored = restoreState({version: 1, config: cfg, phase: 'running', frame: 12, score: 1,
    rngState: 7, bird: {x: 120, y: 300, vy: 0.5},
    pipes: [{id: 1, x: 45, gapY: 200, passed: false}, {id: 2, x: 46, gapY: 200, passed: false}]});
  const scored2 = tick(scored, {});
  assert.equal(scored2.score, 2, 'only the strictly-left pipe (right edge 107 < 108) passes');
  assert.equal(scored2.pipes[0].passed, true);
  assert.equal(scored2.pipes[1].passed, false);
  // Manual colliding snapshot: boundary hit on the next tick awards nothing.
  const manual = restoreState({version: 1, config: cfg, phase: 'running', frame: 5, score: 2,
    rngState: 7, bird: {x: 120, y: 11, vy: 0.5}, pipes: []});
  const over = tick(manual, {});
  assert.equal(over.phase, 'gameover');
  assert.equal(over.score, 2, 'no award on the colliding frame');
  // Pipe overlap collision likewise ends the run without scoring.
  const near = restoreState({version: 1, config: cfg, phase: 'running', frame: 6, score: 3,
    rngState: 7, bird: {x: 120, y: 300, vy: 0.5}, pipes: [{id: 1, x: 100, gapY: 50, passed: false}]});
  const over2 = tick(near, {});
  assert.equal(over2.phase, 'gameover');
  assert.equal(over2.score, 3);
  // Offscreen pipes (right edge < 0) are removed while running.
  const left = restoreState({version: 1, config: cfg, phase: 'running', frame: 8, score: 4,
    rngState: 7, bird: {x: 120, y: 300, vy: 0.5},
    pipes: [{id: 1, x: -70, gapY: 200, passed: true}, {id: 2, x: 250, gapY: 200, passed: false}]});
  const afterLeft = tick(left, {});
  assert.equal(afterLeft.phase, 'running');
  assert.deepEqual(afterLeft.pipes.map((p) => p.id), [2]);
});

test('restoreState validates the complete shape atomically and returns detached states', () => {
  const g = createGame({seed: 5});
  const ok = restoreState(g);
  assert.deepEqual(ok, g);
  assert.notEqual(ok, g);
  assert.notEqual(ok.bird, g.bird);
  assert.notEqual(ok.config, g.config);
  g.bird.y = 5;
  assert.notEqual(ok.bird.y, 5);
  const broken = cloneState(ok);
  broken.bird.y = Number.POSITIVE_INFINITY;
  assert.throws(() => restoreState(broken));
  for (const [key, value] of [
    ['version', 2], ['phase', 'flying'], ['frame', -1], ['score', 1.5], ['rngState', 0], ['pipes', {}]
  ]) {
    const bad = cloneState(ok);
    bad[key] = value;
    assert.throws(() => restoreState(bad), undefined, `${key}=${value}`);
  }
  const dup = cloneState(ok);
  dup.pipes = [{id: 1, x: 480, gapY: 200, passed: false}, {id: 1, x: 100, gapY: 210, passed: true}];
  assert.throws(() => restoreState(dup));
  const extra = cloneState(ok);
  extra.debug = 1;
  assert.throws(() => restoreState(extra));
  assert.throws(() => restoreState({...ok, bird: {x: 1, y: 2}}));
  assert.throws(() => restoreState(null));
  assert.throws(() => restoreState({}));
});

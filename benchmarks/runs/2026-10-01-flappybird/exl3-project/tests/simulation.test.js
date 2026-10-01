import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createGame, tick, flap, pause, resume, restart, cloneState, restoreState,
} from '../src/simulation.js';

test('transitions are pure and match the fixed-step physics', () => {
  const g = createGame();
  assert.equal(g.phase, 'ready');
  assert.deepEqual(g.bird, {x: 120, y: (720 - 60) / 2, vy: 0});
  assert.equal(g.frame, 0);
  assert.ok(Number.isInteger(g.rngState) && g.rngState >= 1);

  const before = JSON.stringify(g);
  const running = flap(g);
  assert.equal(running.phase, 'running');
  assert.equal(running.bird.vy, -6);
  assert.equal(g.phase, 'ready');
  assert.deepEqual(JSON.parse(before), g);

  const t1 = tick(running);
  assert.equal(t1.frame, 1);
  assert.equal(t1.bird.vy, -6 + 0.35);
  assert.equal(t1.bird.y, 330 - 5.65);
  assert.deepEqual(t1.pipes, []);
  assert.deepEqual(running, flap(g));

  // flap inside tick
  const t2 = tick(running, {flap: true});
  assert.equal(t2.bird.vy, -6 + 0.35);
  const paused = pause(running);
  assert.equal(paused.phase, 'paused');
  const untouched = tick(paused, {flap: true});
  assert.equal(untouched.phase, 'paused');
  assert.equal(untouched.bird.y, 330);
  const resumed = resume(paused);
  assert.equal(resumed.phase, 'running');
  assert.equal(resume(running).phase, 'running');
  const r = restart(running);
  assert.equal(r.frame, 0);
  assert.equal(r.phase, 'ready');
  assert.deepEqual(cloneState(r), r);
});

test('spawning, scoring, removal and collision are deterministic', () => {
  const cfg = {gravity: 0.01, spawnEvery: 50};
  const base = createGame(cfg);
  const mk = () => ({...base, phase: 'running', bird: {x: 120, y: 330, vy: 0}});
  let a = mk();
  for (let i = 0; i < 100; i++) a = tick(a);
  let b = mk();
  for (let i = 0; i < 100; i++) b = tick(b);
  assert.equal(a.pipes.length, 2);
  assert.deepEqual(a.pipes.map(p => [p.id, p.x, p.gapY, p.passed]),
    b.pipes.map(p => [p.id, p.x, p.gapY, p.passed]));
  for (const p of a.pipes) {
    assert.ok(p.gapY >= 20 && p.gapY + 180 <= 720 - 60 - 20);
  }
  assert.equal(a.pipes[0].id, 50);
  assert.equal(a.pipes[0].x, 380);
  assert.equal(a.pipes[1].id, 100);
  assert.equal(a.pipes[1].x, 480);
  assert.notEqual(a.rngState, createGame(cfg).rngState);

  // scoring via manually placed pipe behind the bird, gap covering the bird
  const s0 = restoreState({
    version: 1, config: cfg, phase: 'running', frame: 1, score: 0,
    rngState: 0x6d2b79f5, bird: {x: 120, y: 330, vy: 0},
    pipes: [{id: 1, x: 30, gapY: 300, passed: false}],
  });
  const s1 = tick(s0);
  assert.equal(s1.score, 1);
  assert.ok(s1.pipes[0].passed);
  assert.equal(s1.pipes.length, 1); // right edge 28+64=92 >= 0, kept

  // collision frame: pipe in front hits, and a passed-behind pipe must not score
  const c0 = restoreState({
    version: 1, config: cfg, phase: 'running', frame: 1, score: 0,
    rngState: 0x6d2b79f5, bird: {x: 120, y: 330, vy: 0},
    pipes: [
      {id: 1, x: 30, gapY: 300, passed: false},
      {id: 2, x: 128, gapY: 0, passed: false},
    ],
  });
  const c1 = tick(c0);
  assert.equal(c1.phase, 'gameover');
  assert.equal(c1.score, 0);
  // ground boundary
  const g0 = restoreState({
    version: 1, config: cfg, phase: 'running', frame: 1, score: 0,
    rngState: 0x6d2b79f5, bird: {x: 120, y: 648, vy: 1}, pipes: [],
  });
  assert.equal(tick(g0).phase, 'gameover');
});

test('restoreState validates atomically and returns detached copies', () => {
  const snap = {
    version: 1,
    config: {width: 480, height: 720, birdX: 120, birdRadius: 12, gravity: 0.35,
      flapVelocity: -6, pipeSpeed: 2, pipeWidth: 64, gapHeight: 180,
      spawnEvery: 100, groundHeight: 60, seed: 1},
    phase: 'gameover', frame: 5, score: 2, rngState: 7,
    bird: {x: -5, y: 10, vy: 123.456},
    pipes: [{id: 3, x: 10, gapY: 20, passed: true}],
  };
  const restored = restoreState(snap);
  assert.deepEqual(restored.bird, {x: -5, y: 10, vy: 123.456});
  restored.pipes[0].x = 999;
  restored.config.width = 1;
  restored.bird.y = 999;
  assert.equal(snap.pipes[0].x, 10);
  assert.equal(snap.config.width, 480);
  assert.equal(snap.bird.y, 10);

  const bad = [
    null,
    {version: 2},
    {...snap, phase: 'flying'},
    {...snap, frame: 1.5},
    {...snap, score: -1},
    {...snap, rngState: 0},
    {...snap, rngState: 0x100000000},
    {...snap, bird: {x: 1, y: NaN, vy: 0}},
    {...snap, pipes: [{id: 1, x: 0, gapY: 0, passed: true}, {id: 1, x: 5, gapY: 0, passed: false}]},
    {...snap, pipes: [{id: -1, x: 0, gapY: 0, passed: true}]},
    {...snap, pipes: [{id: 1, x: Infinity, gapY: 0, passed: true}]},
    {...snap, pipes: [{id: 1, x: 0, gapY: 0, passed: 'yes'}]},
    {...snap, config: {width: 100}},
  ];
  for (const s of bad) {
    assert.throws(() => restoreState(s));
  }
});

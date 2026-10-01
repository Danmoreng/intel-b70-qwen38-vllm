import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createGame,
  tick,
  flap,
  pause,
  resume,
  restart,
  cloneState,
  restoreState
} from '../src/simulation.js';

test('transitions are pure and return detached states', () => {
  const s0 = createGame({});
  const s1 = flap(s0);
  assert.equal(s1.phase, 'running');
  const input = {flap: true};
  const s2 = tick(s1, input);
  assert.equal(s2.frame, 1);
  assert.equal(s2.bird.vy, -6 + 0.35);
  assert.equal(s1.phase, 'running');
  assert.equal(s1.frame, 0);
  assert.equal(s1.bird.vy, -6);
  assert.deepEqual(input, {flap: true});
  const cloned = cloneState(s2);
  cloned.pipes.push({id: 1});
  cloned.bird.y = 1;
  assert.notDeepEqual(s2, cloned);
});

test('pipes spawn on spawnEvery frames, score once, then are removed', () => {
  const config = {
    pipeSpeed: 10,
    spawnEvery: 10,
    gravity: 0.01,
    flapVelocity: -0.1,
    gapHeight: 620
  };
  let s = flap(createGame(config));
  s = tick(s, {flap: true});
  assert.equal(s.frame, 1);
  assert.equal(s.pipes.length, 0);
  for (let i = 0; i < 9; i += 1) s = tick(s, {flap: true});
  assert.equal(s.frame, 10);
  assert.equal(s.pipes.length, 1);
  const pipe = s.pipes[0];
  assert.equal(pipe.id, 10);
  assert.equal(pipe.x, 480);
  assert.equal(pipe.gapY, 20);
  assert.equal(pipe.passed, false);
  for (let i = 0; i < 44; i += 1) s = tick(s, {flap: true});
  assert.equal(s.phase, 'running');
  assert.equal(s.pipes[0].id, 10);
  assert.equal(s.pipes[0].passed, true);
  assert.equal(s.score, 1);
  for (let i = 0; i < 11; i += 1) s = tick(s, {flap: true});
  assert.ok(s.pipes.every((p) => p.id !== 10));
  assert.equal(s.score, 2);
});

test('collision ends game without scoring; restoreState is strict', () => {
  const base = flap(createGame({}));
  const placed = restoreState({
    version: 1,
    config: base.config,
    phase: 'running',
    frame: 3,
    score: 0,
    rngState: base.rngState,
    bird: {x: 120, y: 330, vy: 0},
    pipes: [{id: 7, x: 120, gapY: 10, passed: false}]
  });
  const s = tick(placed, {});
  assert.equal(s.phase, 'gameover');
  assert.equal(s.score, 0);
  assert.equal(tick(s, {flap: true}).phase, 'gameover');

  let p = pause(flap(createGame({})));
  assert.equal(p.phase, 'paused');
  assert.equal(pause(p).phase, 'paused');
  assert.equal(resume(p).phase, 'running');
  assert.equal(resume(flap(createGame({}))).phase, 'running');
  const again = restart(s);
  assert.equal(again.phase, 'ready');
  assert.equal(again.frame, 0);
  assert.equal(again.rngState, s.rngState);

  assert.throws(() =>
    restoreState({...placed, pipes: [placed.pipes[0], placed.pipes[0]]})
  );
  assert.throws(() => restoreState({...placed, rngState: 0}));
  assert.throws(() => restoreState({...placed, phase: 'flying'}));
});

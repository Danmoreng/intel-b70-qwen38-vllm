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

test('deterministic run, spawn, scoring, collision', () => {
  const cfg = {width: 480, height: 720, birdX: 120, birdRadius: 12, gravity: 0.35, flapVelocity: -6, pipeSpeed: 2, pipeWidth: 64, gapHeight: 180, spawnEvery: 10, groundHeight: 60, seed: 7};
  const a = createGame(cfg);
  const b = createGame(cfg);
  let sa = flap(a);
  let sb = flap(b);
  const gapYs = [];
  for (let i = 0; i < 50; i++) {
    sa = tick(sa, i % 7 === 0 ? {flap: true} : {});
    sb = tick(sb, i % 7 === 0 ? {flap: true} : {});
    if (sa.pipes.length > gapYs.length) gapYs.push(sa.pipes[sa.pipes.length - 1].gapY);
  }
  assert.deepEqual(sa, sb, 'identical inputs produce identical states');
  assert.equal(gapYs.length, 5);
  for (const g of gapYs) {
    assert.ok(g >= 20 && g <= 720 - 60 - 180 - 40 + 20, `gapY ${g} in range`);
  }
  const restarted = restart(sa);
  assert.equal(restarted.phase, 'ready');
  assert.deepEqual(restarted.pipes, []);
  assert.equal(restarted.score, 0);
});

test('purity: detached results, phase-guarded no-ops', () => {
  const s = createGame({seed: 3});
  const paused = pause(pause(flap(s)));
  assert.equal(paused.phase, 'paused');
  const t1 = tick(paused, {flap: true});
  assert.deepEqual(t1, paused, 'tick on paused is detached unchanged');
  paused.pipes.push({id: 1, x: 10, gapY: 100, passed: false});
  const t2 = tick(paused, {});
  assert.equal(t2.pipes.length, 1, 'input not mutated into result');
  const resumed = resume(t2);
  assert.equal(resumed.phase, 'running');
  const noop = flap(paused);
  assert.equal(noop.phase, 'paused', 'flap on paused is a no-op');
  assert.equal(noop.bird.vy, paused.bird.vy);
  assert.notEqual(noop, paused, 'clones are detached');
  cloneState(s).pipes.length = 99;
  assert.equal(s.pipes.length, 0);
});

test('restoreState accepts valid and rejects invalid atomically', () => {
  const base = createGame({seed: 5});
  const good = {
    ...base,
    pipes: [{id: 99, x: -50, gapY: 300.5, passed: true}, {id: 100, x: 600, gapY: 21, passed: false}],
    bird: {x: 120, y: 300, vy: 12.34}
  };
  const restored = restoreState(good);
  assert.deepEqual(restored, good);
  assert.notEqual(restored, good);
  for (const bad of [
    {rngState: 0},
    {phase: 'flying'},
    {frame: -1},
    {score: 1.5},
    {bird: {x: 1, y: 2}},
    {pipes: [{id: 1, x: 0, gapY: 0, passed: false}, {id: 1, x: 5, gapY: 0, passed: true}]},
    {pipes: [{id: 1, x: NaN, gapY: 0, passed: false}]},
    {pipes: [{id: 1, x: 0, gapY: 0, passed: 'yes'}]},
    {version: 2}
  ]) {
    assert.throws(() => restoreState({...good, ...bad}), `bad: ${JSON.stringify(Object.keys(bad))}`);
  }
});

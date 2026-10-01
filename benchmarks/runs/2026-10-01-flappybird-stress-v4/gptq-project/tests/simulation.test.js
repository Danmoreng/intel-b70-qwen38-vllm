import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createGame, tick, flap, pause, resume, restart, cloneState, restoreState,
} from '../src/simulation.js';
import {createRng} from '../src/random.js';
import {DEFAULT_CONFIG} from '../src/config.js';

const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} ~ ${b}`);

test('createGame/flap/tick physics, pause-resume-restart, purity', () => {
  const g = createGame();
  assert.deepEqual(g, {
    version: 1, config: DEFAULT_CONFIG, phase: 'ready', frame: 0, score: 0,
    rngState: 12345, bird: {x: 120, y: 330, vy: 0}, pipes: [],
  });
  assert.notEqual(g.bird, cloneState(g).bird, 'cloneState is a deep clone');

  const s1 = flap(g); // ready -> running
  assert.equal(g.phase, 'ready'); // input untouched
  assert.equal(s1.phase, 'running');
  assert.equal(s1.bird.vy, -6);

  const s2 = tick(s1); // no flap: vy=-6+0.35, y=330-5.65
  assert.equal(s2.frame, 1);
  close(s2.bird.vy, -5.65);
  close(s2.bird.y, 324.35);

  const s3 = tick(s2, {flap: true}); // flap applied before gravity
  assert.equal(s3.frame, 2);
  close(s3.bird.vy, -5.65);
  close(s3.bird.y, 318.7);

  const p = pause(s3);
  assert.equal(p.phase, 'paused');
  assert.deepEqual(pause(p), p, 'pause on paused is a detached no-op');
  const frozen = tick(p, {flap: true}); // nonrunning: detached unchanged
  assert.deepEqual(frozen, p);
  assert.equal(p.frame, 2);
  assert.deepEqual(flap(p), p, 'flap on paused is a no-op');
  assert.equal(resume(p).phase, 'running');
  assert.equal(tick(resume(p)).frame, 3);
  assert.equal(resume(s1).phase, 'running', 'resume on running is a no-op');
  const r = restart(s3);
  assert.deepEqual(r, g, 'restart resets to the fresh ready state of same config');
});

test('deterministic spawns, scoring and rngState under a fast-spawn config', () => {
  const g = createGame({spawnEvery: 10, pipeSpeed: 5, birdX: 460, gapHeight: 420});
  let cur = flap(g);
  for (let i = 1; i <= 19; i += 1) cur = tick(cur); // frames 1..19
  cur = tick(cur, {flap: true}); // frame 20: mid-air flap, then frames 21..30
  for (let i = 0; i < 10; i += 1) cur = tick(cur);
  assert.equal(cur.frame, 30);
  assert.equal(cur.phase, 'running');
  assert.equal(cur.score, 1); // only id-10: right edge 444 < 448 (=460-12); id-20 edge 494
  assert.deepEqual(cur.pipes.map((p) => p.id), [10, 20, 30]);
  assert.deepEqual(cur.pipes.map((p) => p.x), [380, 430, 480]);
  assert.deepEqual(cur.pipes.map((p) => p.passed), [true, false, false]);
  assert.ok(cur.pipes.every((p) => Number.isInteger(p.gapY) && p.gapY >= 20 && p.gapY <= 220));
  assert.notEqual(cur.rngState, 12345, 'rngState advanced by the three spawns');
  assert.ok(Number.isInteger(cur.rngState) && cur.rngState > 0 && cur.rngState <= 0xffffffff);
  // re-deriving the next spawn (frame 40) from the saved rngState gives its gapY
  const span = 720 - 60 - 420 - 40 + 1;
  const nextGapY = 20 + Math.floor(createRng(cur.rngState).next() * span);
  for (let f = 31; f <= 40; f += 1) cur = tick(cur);
  assert.equal(cur.frame, 40);
  assert.equal(cur.phase, 'running');
  const newest = cur.pipes[cur.pipes.length - 1];
  assert.equal(newest.id, 40);
  assert.equal(newest.gapY, nextGapY);
  assert.equal(newest.x, 480, 'spawns at x=width, un-moved on its own frame');
});

test('restoreState round-trip/detachment, collision, boundary, removal, rejection', () => {
  const snap = {
    version: 1,
    config: {...DEFAULT_CONFIG},
    phase: 'running',
    frame: 5,
    score: 2,
    rngState: createRng(12345).uint32(), // any nonzero uint32
    bird: {x: 120, y: 330, vy: 0},
    pipes: [
      {id: 1, x: 40, gapY: 200, passed: false}, // right edge 104 < 108 -> scores, stays
      {id: 2, x: -66, gapY: 200, passed: false}, // right edge -2 < 0 -> removed
    ],
  };
  const s = restoreState(snap);
  assert.deepEqual(snap.pipes[0], {id: 1, x: 40, gapY: 200, passed: false}, 'snapshot detached');
  snap.bird.y = 999;
  snap.pipes.push({id: 3, x: 9, gapY: 1, passed: true});
  assert.equal(s.bird.y, 330);
  assert.equal(s.pipes.length, 2);

  const s1 = tick(s);
  assert.equal(s1.frame, 6);
  assert.equal(s1.score, 4); // both unpassed pipes cleared the scoring line
  assert.equal(s1.pipes.length, 1); // id 2 removed (right edge < 0)
  assert.equal(s1.pipes[0].id, 1);
  assert.equal(s1.pipes[0].passed, true);

  // pipe collision (inclusive edges) -> gameover, no point awarded that frame
  const snap2 = {
    version: 1, config: {...DEFAULT_CONFIG}, phase: 'running', frame: 5, score: 2,
    rngState: 12345, bird: {x: 120, y: 100, vy: 0}, pipes: [{id: 1, x: 50, gapY: 400, passed: false}],
  };
  const go = tick(restoreState(snap2));
  assert.equal(go.phase, 'gameover');
  assert.equal(go.score, 2);
  assert.equal(go.pipes[0].passed, false);
  assert.deepEqual(tick(go), go, 'tick on gameover is a detached no-op');
  assert.deepEqual(flap(go), go, 'flap on gameover is a no-op');

  const ground = {
    version: 1, config: {...DEFAULT_CONFIG}, phase: 'running', frame: 0, score: 0,
    rngState: 12345, bird: {x: 120, y: 652, vy: 0}, pipes: [],
  };
  assert.equal(tick(restoreState(ground)).phase, 'gameover'); // y+r = 664 >= 660

  // invalid snapshots reject atomically
  const bad = (mut) => {
    const m = cloneState(restoreState(snap));
    mut(m);
    assert.throws(() => restoreState(m));
  };
  bad((m) => m.pipes.push({id: 1, x: 0, gapY: 1, passed: false})); // duplicate id
  bad((m) => { m.rngState = 0; });
  bad((m) => { m.rngState = 0x100000000; });
  bad((m) => { m.bird.x = NaN; });
  bad((m) => { m.bird.vy = Infinity; });
  bad((m) => { m.phase = 'flying'; });
  bad((m) => { m.version = 2; });
  bad((m) => { m.frame = -1; });
  bad((m) => { m.pipes[0].passed = 1; });
  bad((m) => { m.pipes[0].x = NaN; });
});

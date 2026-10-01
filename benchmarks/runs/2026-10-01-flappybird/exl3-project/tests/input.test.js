import test from 'node:test';
import assert from 'node:assert/strict';
import {createInput, normalizePointer, createTicker} from '../src/input.js';

test('edge-triggered input with consume, reset and dispose', () => {
  const input = createInput();
  assert.equal(input.handle({code: 'Space'}), true);
  assert.equal(input.handle({code: 'Space'}), true); // held, no queue
  assert.equal(input.handle({code: 'ArrowUp'}), true);
  assert.equal(input.handle({code: 'KeyP', repeat: true}), true);
  assert.deepEqual(input.consume(), ['flap', 'flap']);
  assert.equal(input.consume().length, 0);
  assert.equal(input.handle({code: 'KeyP'}), true); // first non-repeat while not held
  assert.equal(input.release({code: 'KeyP'}), true);
  assert.equal(input.release({code: 'KeyP'}), true); // recognized even when not held
  assert.equal(input.handle({code: 'KeyP', repeat: true}), true);
  assert.deepEqual(input.consume(), ['pause']);
  assert.equal(input.handle({code: 'Digit9'}), false);
  assert.equal(input.release({code: 'Digit9'}), false);
  assert.equal(input.handle({code: 'Space', type: 'keyup'}), true);
  assert.equal(input.handle({code: 'Space'}), true); // keyup released it
  assert.deepEqual(input.consume(), ['flap']);

  input.handle({code: 'KeyR'});
  input.reset();
  assert.deepEqual(input.consume(), []);
  assert.equal(input.handle({code: 'KeyR'}), true);
  input.dispose();
  assert.equal(input.handle({code: 'Space'}), false);
  assert.equal(input.release({code: 'Space'}), false);
  assert.deepEqual(input.consume(), []);

  assert.throws(() => createInput({bogus: ['KeyA']}));
  assert.throws(() => createInput({flap: ['Space'], pause: ['Space']}));
  assert.throws(() => createInput({flap: ['Space', 'Space']}));
  assert.throws(() => createInput({flap: ['']}));
  assert.throws(() => createInput({flap: 'Space'}));
  assert.equal(createInput({}).handle({code: 'Space'}), false);
});

test('normalizePointer maps and clamps, rejects malformed', () => {
  assert.deepEqual(
    normalizePointer({clientX: 100, clientY: 50}, {left: 50, top: 25, width: 200, height: 100}, 480, 720),
    {x: 120, y: 180},
  );
  assert.deepEqual(
    normalizePointer({clientX: -50, clientY: 1000}, {left: 0, top: 0, width: 100, height: 100}, 480, 720),
    {x: 0, y: 720},
  );
  assert.deepEqual(
    normalizePointer({clientX: 500, clientY: -5}, {left: 0, top: 0, width: 100, height: 100}, 480, 720),
    {x: 480, y: 0},
  );
  assert.throws(() => normalizePointer({clientX: NaN, clientY: 0}, {left: 0, top: 0, width: 1, height: 1}, 10, 10));
  assert.throws(() => normalizePointer({clientX: 0, clientY: 0}, {left: 0, top: 0, width: 0, height: 1}, 10, 10));
  assert.throws(() => normalizePointer({clientX: 0, clientY: 0}, {left: 0, top: 0, width: 1, height: 1}, 10, 0));
  assert.throws(() => normalizePointer(null, {left: 0, top: 0, width: 1, height: 1}, 10, 10));
});

test('ticker: fixed steps, backlog cap, pause/resume, atomic rejects', () => {
  const calls = [];
  const t = createTicker(dt => calls.push(dt), {hz: 100, maxSteps: 3});
  assert.equal(t.update(0), 0); // first timestamp initializes
  assert.equal(t.update(50), 3); // 5 steps available, capped at 3, backlog dropped
  assert.ok(calls.every(dt => Math.abs(dt - 0.01) < 1e-12));
  assert.equal(t.update(100.5), 3);
  assert.equal(calls.length, 6);
  assert.equal(t.update(110), 0); // fresh base after drop, 9.5ms -> 0
  assert.throws(() => t.update(100)); // decreasing rejects atomically
  assert.equal(t.update(115), 1); // state unchanged by reject: 14.5ms -> 1
  t.pause();
  const before = calls.length;
  assert.equal(t.update(20000), 0);
  assert.equal(calls.length, before);
  t.resume();
  assert.equal(t.update(20000), 0); // fresh base
  assert.equal(t.update(20020), 2);
  t.reset();
  assert.equal(t.update(1), 0);
  assert.equal(t.update(11), 1);
  assert.throws(() => t.update(NaN));
  assert.throws(() => createTicker(() => {}, {hz: 0}));
  assert.throws(() => createTicker(() => {}, {hz: 241}));
  assert.throws(() => createTicker(() => {}, {maxSteps: 0}));
  assert.throws(() => createTicker(() => {}, {maxSteps: 101}));
});

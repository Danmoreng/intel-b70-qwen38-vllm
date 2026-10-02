import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createInput, normalizePointer, createTicker} from '../src/input.js';

test('edge-triggered keys: queue once, release, reset, dispose', () => {
  const input = createInput();
  assert.equal(input.handle({code: 'Space'}), true, 'queues flap on first keydown');
  assert.equal(input.handle({code: 'Space'}), true, 'held second keydown recognized');
  assert.equal(input.handle({code: 'Space', repeat: true}), true, 'repeat recognized');
  assert.deepEqual(input.consume(), ['flap']);
  assert.deepEqual(input.consume(), [], 'queue emptied');

  assert.equal(input.handle({code: 'F1'}), false, 'unknown code not recognized');
  assert.equal(input.release({code: 'F1'}), false);

  input.reset();
  assert.deepEqual(input.consume(), []);
  assert.equal(input.handle({code: 'KeyP'}), true);
  assert.equal(input.handle({code: 'ArrowUp'}), true);
  assert.deepEqual(input.consume(), ['pause', 'flap'], 'actions in keydown order');
  input.reset();
  assert.deepEqual(input.consume(), []);
  assert.equal(input.handle({code: 'KeyP'}), true, 'reset clears held set');

  assert.equal(input.handle({code: 'KeyP', type: 'keyup'}), true, 'keyup delegates');
  assert.equal(input.release({code: 'KeyP'}), true, 'recognized even when not held');
  assert.equal(input.handle({code: 'KeyP'}), true, 're-held after release');
  input.dispose();
  assert.equal(input.handle({code: 'Space'}), false);
  assert.equal(input.release({code: 'Space'}), false);
  assert.deepEqual(input.consume(), []);

  assert.throws(() => createInput({jump: ['X']}), Error, 'unknown action');
  assert.throws(() => createInput({flap: ['Space'], pause: ['Space']}), Error, 'dup code');
  assert.throws(() => createInput({flap: ['']}), Error, 'empty code');
  assert.throws(() => createInput({flap: 'Space'}), Error, 'non-array binding');
  assert.throws(() => createInput(null), Error);
});

test('normalizePointer maps and clamps; malformed throws', () => {
  const rect = {left: 5, top: 5, width: 100, height: 200};
  assert.deepEqual(normalizePointer({clientX: 10, clientY: 20}, rect, 100, 200),
    {x: 5, y: 15});
  assert.deepEqual(normalizePointer({clientX: -500, clientY: 5000}, rect, 960, 540),
    {x: 0, y: 540}, 'out-of-bounds clamped to edges');
  assert.throws(() => normalizePointer(null, rect, 10, 10), Error);
  assert.throws(() => normalizePointer({clientX: NaN, clientY: 1}, rect, 10, 10), Error);
  assert.throws(() => normalizePointer({clientX: 1, clientY: 1},
    {left: 0, top: 0, width: 0, height: 10}, 10, 10), Error);
  assert.throws(() => normalizePointer({clientX: 1, clientY: 1},
    {left: 0, top: 0, width: 10, height: 10}, 0, 10), Error);
  assert.throws(() => normalizePointer({clientX: 1, clientY: 1},
    {left: 0, top: 0, width: 10, height: 10}, 10, Infinity), Error);
});

test('ticker: fixed dt, capped backlog, fractional remainder, atomic rejects', () => {
  assert.throws(() => createTicker(() => {}, {hz: 0}), Error);
  assert.throws(() => createTicker(() => {}, {hz: 241}), Error);
  assert.throws(() => createTicker(() => {}, {hz: NaN}), Error);
  assert.throws(() => createTicker(() => {}, {hz: 60, maxSteps: 0}), Error);
  assert.throws(() => createTicker(() => {}, {hz: 60, maxSteps: 101}), Error);
  assert.throws(() => createTicker(() => {}, {hz: 60, maxSteps: 1.5}), Error);

  const dts = [];
  const ticker = createTicker((dt) => dts.push(dt), {hz: 100, maxSteps: 5});
  const update = (t) => {
    ticker.update(t);
    return dts.length;
  };

  assert.equal(update(0), 0, 'first timestamp only initializes');
  assert.equal(update(25), 2, 'two fixed steps within 25ms');
  assert.equal(update(25), 2, 'same timestamp does nothing');
  assert.equal(update(29), 2, 'fractional remainder retained');
  assert.equal(update(30), 3);
  assert.equal(update(130), 8, 'backlog of 100ms capped at maxSteps=5');
  assert.equal(update(130), 8, 'excess backlog (incl. remainder) dropped');
  assert.equal(update(139), 8, 'remainder 9ms < dt');
  assert.equal(update(140), 9, 'remainder 10ms crosses dt');

  assert.throws(() => ticker.update(139), Error, 'decreasing timestamp rejected');
  assert.equal(update(150), 10, 'state intact after rejection');
  assert.throws(() => ticker.update(NaN), Error);
  assert.throws(() => ticker.update(-Infinity), Error);

  ticker.pause();
  assert.equal(update(5000), 10, 'paused update never fires');
  ticker.resume();
  assert.equal(update(5000), 10, 'resume starts a fresh base');
  assert.equal(update(5015), 11);
  ticker.reset();
  assert.equal(update(9000), 11, 'reset takes a fresh base');
  assert.ok(dts.every((dt) => dt === 0.01), 'every callback gets dt = 1000/hz');
  assert.equal(dts.length, 11);
});

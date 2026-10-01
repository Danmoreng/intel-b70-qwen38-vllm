import test from 'node:test';
import assert from 'node:assert/strict';
import {createInput, normalizePointer, createTicker} from '../src/input.js';

test('edge-triggered input with dispose and reset', () => {
  const input = createInput();
  assert.equal(input.handle({code: 'Space'}), true);
  assert.equal(input.handle({code: 'Space'}), true);
  assert.deepEqual(input.consume(), ['flap']);
  assert.equal(input.release({code: 'Space'}), true);
  assert.equal(input.handle({code: 'Space', repeat: true}), true);
  assert.deepEqual(input.consume(), []);

  assert.equal(input.handle({code: 'Nope'}), false);
  assert.equal(input.handle({code: 'KeyP'}), true);
  assert.equal(input.handle({code: 'KeyP', type: 'keyup'}), true);
  assert.deepEqual(input.consume(), ['pause']);
  assert.equal(input.release({code: 'Nope'}), false);

  input.handle({code: 'KeyR'});
  input.reset();
  assert.deepEqual(input.consume(), []);
  assert.equal(input.handle({code: 'KeyR'}), true);
  input.dispose();
  assert.equal(input.handle({code: 'Space'}), false);
  assert.equal(input.release({code: 'Space'}), false);
  assert.deepEqual(input.consume(), []);

  assert.throws(() => createInput({flap: ['Space'], weird: ['X']}));
  assert.throws(() => createInput({flap: ['Space', 'Space']}));
  assert.throws(() => createInput({flap: ['']}));
});

test('normalizePointer maps and clamps, rejects malformed', () => {
  const rect = {left: 10, top: 20, width: 200, height: 100};
  assert.deepEqual(normalizePointer({clientX: 110, clientY: 70}, rect, 480, 720), {x: 240, y: 360});
  assert.deepEqual(normalizePointer({clientX: -1000, clientY: 1e6}, rect, 480, 720), {x: 0, y: 720});
  assert.throws(() => normalizePointer({clientX: NaN, clientY: 0}, rect, 480, 720));
  assert.throws(() => normalizePointer({clientX: 0, clientY: 0}, {left: 0, top: 0, width: 0, height: 100}, 480, 720));
  assert.throws(() => normalizePointer({clientX: 0, clientY: 0}, rect, -1, 720));
});

test('bounded fixed-step ticker', () => {
  const calls = [];
  const ticker = createTicker((dt) => calls.push(dt), {hz: 100, maxSteps: 3});
  ticker.update(0);
  assert.equal(calls.length, 0);
  ticker.update(21);
  assert.equal(calls.length, 2);
  assert.ok(calls.every((dt) => Math.abs(dt - 0.01) < 1e-12));
  ticker.update(70);
  assert.equal(calls.length, 5);
  ticker.update(72);
  assert.equal(calls.length, 5);
  ticker.update(110);
  assert.equal(calls.length, 8);

  ticker.pause();
  ticker.update(1000);
  assert.equal(calls.length, 8);
  ticker.resume();
  ticker.update(200);
  assert.equal(calls.length, 8);
  ticker.update(210);
  assert.equal(calls.length, 9);

  assert.throws(() => ticker.update(1));
  assert.throws(() => ticker.update(NaN));
  assert.throws(() => createTicker(() => {}, {hz: 241}));
  assert.throws(() => createTicker(() => {}, {maxSteps: 101}));
  assert.throws(() => createTicker('nope'));

  const fresh = createTicker((dt) => calls.push(dt));
  fresh.update(500);
  fresh.reset();
  fresh.update(500);
  fresh.update(500 + 1000 / 60);
  assert.equal(calls.length, 10);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import {createInput, normalizePointer, createTicker} from '../src/input.js';

test('edge-triggered keys, releases, reset, dispose, bindings', () => {
  const input = createInput();
  assert.equal(input.handle({code: 'Space'}), true);
  assert.equal(input.handle({code: 'Space'}), true, 'second held keydown recognized, no queue');
  assert.equal(input.consume().length, 1);
  assert.deepEqual(input.consume(), []);
  assert.equal(input.release({code: 'Space'}), true);
  assert.equal(input.handle({code: 'Space', repeat: true}), true, 'repeat recognized, no queue');
  assert.deepEqual(input.consume(), []);
  assert.equal(input.handle({code: 'Space', type: 'keyup'}), true, 'type keyup delegates to release');
  assert.equal(input.handle({code: 'KeyZ'}), false);
  assert.equal(input.handle({code: 'ArrowUp'}), true);
  assert.equal(input.handle({code: 'KeyP'}), true);
  assert.deepEqual(input.consume(), ['flap', 'pause']);
  input.reset();
  assert.deepEqual(input.consume(), []);
  input.handle({code: 'KeyR'});
  input.dispose();
  assert.equal(input.handle({code: 'Space'}), false);
  assert.equal(input.release({code: 'KeyR'}), false);
  assert.deepEqual(input.consume(), []);
  const custom = createInput({flap: ['Space']});
  assert.equal(custom.handle({code: 'KeyP'}), false, 'unbound when custom bindings given');
  for (const bad of [{jump: ['Space']}, {flap: ['']}, {flap: ['Space'], pause: ['Space']}, {flap: 'Space'}]) {
    assert.throws(() => createInput(bad), `bad bindings ${JSON.stringify(bad)}`);
  }
});

test('normalizePointer maps, clamps, rejects malformed', () => {
  assert.deepEqual(normalizePointer({clientX: 100, clientY: 100}, {left: 0, top: 0, width: 200, height: 200}, 480, 720), {x: 240, y: 360});
  assert.deepEqual(normalizePointer({clientX: -10, clientY: 5000}, {left: 0, top: 0, width: 100, height: 100}, 10, 10), {x: 0, y: 10});
  assert.equal(normalizePointer({clientX: 30, clientY: 40}, {left: 10, top: 20, width: 50, height: 50}, 200, 200).x, 80);
  for (const [ev, rect, w, h] of [
    [{clientX: NaN, clientY: 0}, {left: 0, top: 0, width: 1, height: 1}, 1, 1],
    [{clientX: 0, clientY: 0}, {left: 0, top: 0, width: 0, height: 1}, 1, 1],
    [{clientX: 0, clientY: 0}, {left: 0, top: 0, width: 1, height: 1}, 0, 1]
  ]) {
    assert.throws(() => normalizePointer(ev, rect, w, h));
  }
});

test('fixed-step ticker: init, backlog drop, pause/resume/reset, monotonic', () => {
  const calls = [];
  const ticker = createTicker((t) => calls.push(t), {hz: 100, maxSteps: 5});
  ticker.update(0);
  assert.deepEqual(calls, []);
  ticker.update(100); // 10ms * 10, capped at 5
  assert.equal(calls.length, 5);
  ticker.update(125); // remainder 5 dropped, +25 -> 2 steps
  assert.equal(calls.length, 7);
  assert.equal(calls[0], 0.01);
  ticker.pause();
  ticker.update(500);
  assert.equal(calls.length, 7, 'paused updates never step');
  ticker.resume();
  ticker.update(1000);
  assert.equal(calls.length, 7, 'resume starts fresh base');
  ticker.update(1020);
  assert.equal(calls.length, 9);
  assert.throws(() => ticker.update(1010), 'decreasing timestamp rejected');
  assert.throws(() => ticker.update(NaN), 'nonfinite timestamp rejected');
  ticker.update(1020);
  assert.equal(calls.length, 9, 'equal timestamp steps nothing');
  ticker.reset();
  ticker.update(10000);
  ticker.update(10000);
  assert.equal(calls.length, 9, 'reset starts fresh base');
  for (const bad of [{hz: 0}, {hz: 241}, {hz: NaN}, {maxSteps: 0}, {maxSteps: 101}, {maxSteps: 2.5}]) {
    assert.throws(() => createTicker(() => {}, bad));
  }
});

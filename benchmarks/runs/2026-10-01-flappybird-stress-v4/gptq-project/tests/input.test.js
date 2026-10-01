import test from 'node:test';
import assert from 'node:assert/strict';
import {createInput, normalizePointer, createTicker} from '../src/input.js';

test('edge-triggered bindings: repeat/second-keydown ignored, release/keyup/reset/dispose', () => {
  const input = createInput();
  assert.equal(input.handle({code: 'Space'}), true);
  assert.equal(input.handle({code: 'Space'}), true, 'held: recognized, no second edge');
  assert.deepEqual(input.consume(), ['flap']);
  assert.equal(input.release({code: 'Space'}), true);
  assert.equal(input.release({code: 'Space'}), true); // recognized even when not held
  assert.equal(input.handle({code: 'Space', repeat: true}), true, 'repeat: recognized, no edge');
  assert.deepEqual(input.consume(), [], 'repeats queue nothing');
  assert.equal(input.handle({code: 'ArrowUp'}), true);
  assert.equal(input.handle({code: 'ArrowUp', type: 'keyup'}), true, 'keyup delegates to release');
  assert.deepEqual(input.consume(), ['flap']);
  assert.deepEqual(input.consume(), [], 'consume empties');
  assert.equal(input.handle({code: 'KeyP'}), true);
  assert.equal(input.handle({code: 'KeyR'}), true);
  assert.deepEqual(input.consume(), ['pause', 'restart'], 'actions queued in order');
  assert.equal(input.handle({code: 'KeyX'}), false, 'unrecognized code');
  assert.equal(input.handle({code: 'KeyP', type: 'keypress'}), false, 'other types ignored');
  assert.equal(input.release({code: 'KeyX'}), false, 'unrecognized release');
  input.handle({code: 'Space'});
  input.reset();
  assert.deepEqual(input.consume(), []);
  assert.equal(input.handle({code: 'Space'}), true, 'held cleared by reset -> fresh edge');
  input.dispose();
  assert.equal(input.handle({code: 'Space'}), false);
  assert.equal(input.release({code: 'Space'}), false);
  assert.deepEqual(input.consume(), []);

  assert.throws(() => createInput({flap: ['Space'], pause: ['Space']}));
  assert.throws(() => createInput({flap: ['Space', 'Space']}));
  assert.throws(() => createInput({jump: ['Space']}));
  assert.throws(() => createInput({flap: ['Space', '']}));
  assert.throws(() => createInput({flap: 'Space'}));
  assert.throws(() => createInput(null));
});

test('normalizePointer clamps to logical space and rejects malformed input', () => {
  const rect = {left: 10, top: 20, width: 100, height: 50};
  assert.deepEqual(normalizePointer({clientX: 60, clientY: 45}, rect, 480, 720), {x: 240, y: 360});
  assert.deepEqual(normalizePointer({clientX: 10, clientY: 20}, rect, 480, 720), {x: 0, y: 0});
  assert.deepEqual(normalizePointer({clientX: 110, clientY: 70}, rect, 480, 720), {x: 480, y: 720});
  assert.deepEqual(normalizePointer({clientX: -50, clientY: 500}, rect, 480, 720), {x: 0, y: 720});
  assert.throws(() => normalizePointer({clientX: NaN, clientY: 1}, rect, 480, 720));
  assert.throws(() => normalizePointer({clientX: 1, clientY: 1}, {left: 0, top: 0, width: 0, height: 10}));
  assert.throws(() => normalizePointer({clientX: 1, clientY: 1}, {left: 0, top: 0, width: -2, height: 10}));
  assert.throws(() => normalizePointer({clientX: 1, clientY: 1}, null, 480, 720));
  assert.throws(() => normalizePointer({clientX: 1, clientY: 1}, rect, 0, 720));
  assert.throws(() => normalizePointer({clientX: 1, clientY: 1}, rect, 480, Infinity));
});

test('createTicker: init, fixed steps, backlog cap with remainder, pause/resume/reset, validation', () => {
  const calls = [];
  const t = createTicker((s) => calls.push(s), {hz: 100, maxSteps: 4}); // dt is 1/100 s
  assert.equal(t.update(0), 0, 'first timestamp only initializes');
  assert.deepEqual(calls, []);
  assert.equal(t.update(4), 0, 'sub-step elapsed: no callback, window base kept');
  assert.equal(t.update(40), 4, 'retained remainder plus full windows run to the cap');
  assert.equal(t.update(50), 1);
  assert.equal(calls.length, 5);
  assert.ok(calls.every((s) => Math.abs(s - 0.01) < 1e-12), 'step dt equals 1 over hz');
  assert.throws(() => t.update(40), RangeError, 'decreasing timestamp rejects atomically');
  assert.equal(calls.length, 5, 'no state change from the rejected timestamp');
  assert.equal(t.update(500), 4, 'excess backlog dropped at cap');
  assert.equal(t.update(510), 1, 'fresh base after backlog drop');
  assert.throws(() => t.update(NaN), RangeError, 'nonfinite timestamps rejected');
  assert.throws(() => t.update(Infinity), RangeError);
  assert.throws(() => t.update(-1), RangeError, 'decreasing timestamps reject atomically');
  assert.equal(t.update(510), 0, 'rejections left no trace');
  assert.equal(t.update(520), 1);
  assert.equal(calls.length, 11);
  t.pause();
  assert.equal(t.update(6000), 0, 'paused: no callbacks, base cleared');
  assert.equal(calls.length, 11);
  t.resume();
  assert.equal(t.update(6000), 0, 'resume starts fresh base');
  assert.equal(t.update(6010), 1);
  t.reset();
  assert.equal(t.update(7000), 0, 'reset gives a fresh base');
  assert.equal(t.update(7010), 1);
  assert.equal(calls.length, 13);
  assert.throws(() => createTicker(() => {}, {hz: 0}));
  assert.throws(() => createTicker(() => {}, {hz: 241}));
  assert.throws(() => createTicker(() => {}, {hz: NaN}));
  assert.throws(() => createTicker(() => {}, {maxSteps: 0}));
  assert.throws(() => createTicker(() => {}, {maxSteps: 101}));
  assert.throws(() => createTicker(() => {}, {maxSteps: 2.5}));
});

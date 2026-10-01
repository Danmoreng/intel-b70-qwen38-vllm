import test from 'node:test';
import assert from 'node:assert/strict';
import {createRng, hashSeed} from '../src/random.js';

test('separate instances stay independent', () => {
  const a = createRng(7);
  const b = createRng(7);
  const first = a.uint32();
  a.uint32();
  assert.notEqual(a.snapshot(), b.snapshot());
  assert.equal(b.uint32(), first);
});

test('seed rejects non-uint32 values', () => {
  assert.throws(() => createRng(-1), TypeError);
  assert.throws(() => createRng(1.5), TypeError);
  assert.throws(() => createRng('7'), TypeError);
  assert.throws(() => createRng(2 ** 32), TypeError);
});

test('hashSeed is deterministic unsigned FNV-1a', () => {
  assert.equal(hashSeed(''), 2166136261);
  assert.equal(typeof hashSeed('flappy'), 'number');
  assert.equal(hashSeed('flappy'), hashSeed('flappy'));
  assert.ok(hashSeed('flappy') >= 0 && Number.isInteger(hashSeed('flappy')));
});

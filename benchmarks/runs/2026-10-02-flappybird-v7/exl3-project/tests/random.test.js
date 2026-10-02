import test from 'node:test';
import assert from 'node:assert/strict';
import {createRng, hashSeed} from '../src/random.js';

test('reproducible sequences and separate instances', () => {
  const a = createRng(42);
  const b = createRng(42);
  assert.deepEqual(
    [a.uint32(), a.uint32(), a.next()],
    [b.uint32(), b.uint32(), b.next()]
  );
  a.uint32();
  assert.notEqual(a.snapshot(), b.snapshot());
});

test('hashSeed edge cases', () => {
  assert.equal(hashSeed(''), 2166136261);
  assert.equal(hashSeed('a'), Math.imul(2166136261 ^ 97, 16777619) >>> 0);
  assert.throws(() => hashSeed(1), TypeError);
});

test('seed and restore rejection', () => {
  for (const bad of [NaN, Infinity, -1, 0.5, 2 ** 32, '7', null]) {
    assert.throws(() => createRng(bad));
  }
  const r = createRng(5);
  for (const bad of [0, -1, 1.5, 2 ** 32, '5', undefined]) {
    assert.throws(() => r.restore(bad));
  }
  assert.equal(r.snapshot(), 5, 'invalid restore must not change state');
  r.restore(0xffffffff);
  assert.equal(r.snapshot(), 0xffffffff);
});

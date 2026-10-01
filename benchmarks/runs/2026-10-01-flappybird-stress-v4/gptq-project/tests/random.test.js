import test from 'node:test';
import assert from 'node:assert/strict';
import {createRng, hashSeed} from '../src/random.js';

const MASK = 0xffffffffn;
function referenceSequence(seed, count) {
  let x = (seed === 0 ? 0x6d2b79f5n : BigInt(seed)) & MASK;
  const out = [];
  for (let i = 0; i < count; i += 1) {
    x = (x ^ (x << 13n)) & MASK;
    x = (x ^ (x >> 17n)) & MASK;
    x = (x ^ (x << 5n)) & MASK;
    out.push(Number(x));
  }
  return out;
}

test('xorshift32 sequence matches independent reference; zero seed init', () => {
  for (const seed of [1, 7, 12345, 4294967295, 0]) {
    const rng = createRng(seed);
    assert.equal(rng.snapshot(), seed === 0 ? 0x6d2b79f5 : seed);
    const expected = referenceSequence(seed, 5);
    const got = Array.from({length: 5}, () => rng.uint32());
    assert.deepEqual(got, expected, `seed ${seed}`);
    // next() is uint32()/2^32 in [0,1)
    const n = rng.next();
    assert.ok(n >= 0 && n < 1 && Number.isFinite(n));
    assert.equal(n, rng.snapshot() / 4294967296);
  }
});

test('instances are independent and snapshot/restore round-trips', () => {
  const a = createRng(99);
  const b = createRng(99);
  const s0 = a.snapshot();
  assert.equal(s0, b.snapshot());
  const run1 = Array.from({length: 3}, () => a.uint32());
  const run2 = Array.from({length: 3}, () => b.uint32());
  assert.deepEqual(run1, run2);
  a.restore(s0);
  assert.deepEqual(Array.from({length: 3}, () => a.uint32()), run1);
  // invalid restores throw and never change state
  const before = a.snapshot();
  for (const bad of [0, -1, 4294967296, 1.5, '5', null, undefined, true, []]) {
    assert.throws(() => a.restore(bad));
    assert.equal(a.snapshot(), before);
  }
  assert.equal(a.restore(before), undefined);
  assert.equal(a.snapshot(), before);
});

test('invalid seeds are rejected; hashSeed is FNV-1a over UTF-16 code units', () => {
  for (const bad of [-1, 2 ** 32, 1.5, NaN, Infinity, '123', true, null, [], undefined]) {
    assert.throws(() => createRng(bad));
  }
  assert.equal(hashSeed(''), 2166136261); // FNV offset basis
  assert.equal(hashSeed('a'), 3826002220); // 0x811c9dc5 ^ 97, then imul by FNV prime
  // Reference walk over explicit UTF-16 code units (surrogate pair hashed as two units)
  const s = 'A\uD83D\uDE00B';
  const units = [];
  for (let i = 0; i < s.length; i += 1) units.push(s.charCodeAt(i));
  assert.deepEqual(units, [0x41, 0xd83d, 0xde00, 0x42]);
  let h = 2166136261;
  for (const cu of [0x41, 0xd83d, 0xde00, 0x42]) {
    h = Math.imul(h ^ cu, 16777619) >>> 0;
  }
  assert.equal(hashSeed(s), h);
  // hashing by code point instead of code units must give a different result
  let hcp = 2166136261;
  hcp = Math.imul(hcp ^ 0x41, 16777619) >>> 0;
  hcp = Math.imul(hcp ^ 0x1f600, 16777619) >>> 0;
  hcp = Math.imul(hcp ^ 0x42, 16777619) >>> 0;
  assert.notEqual(hashSeed(s), hcp);
});

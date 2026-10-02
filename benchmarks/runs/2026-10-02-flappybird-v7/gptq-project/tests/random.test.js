import { test } from 'node:test';
import assert from 'node:assert/strict';
import {createRng, hashSeed} from '../src/random.js';

const MASK = 0xffffffffn;

// Independent reference: FNV-1a with BigInt so no shared 32-bit code path.
function refHash(text) {
  let hash = 2166136261n;
  for (let i = 0; i < text.length; i += 1) {
    hash = ((hash ^ BigInt(text.charCodeAt(i))) * 16777619n) % (2n ** 32n);
  }
  return Number(hash);
}

test('zero seed initializes to 0x6d2b79f5; snapshot/restore round trips', () => {
  const rng = createRng(0);
  assert.equal(rng.snapshot(), 0x6d2b79f5);
  const first = rng.uint32();
  assert.equal(rng.snapshot(), first, 'snapshot is the latest generated state');
  rng.restore(0x6d2b79f5);
  assert.equal(rng.uint32(), first, 'restore rewinds to the initial state');

  const a = createRng(12345);
  const b = createRng(12345);
  const seqA = Array.from({length: 5}, () => a.uint32());
  const seqB = Array.from({length: 5}, () => b.uint32());
  assert.deepEqual(seqA, seqB, 'separate instances, same seed, same sequence');

  const snap = a.snapshot();
  const extra = Array.from({length: 3}, () => a.uint32());
  a.restore(snap);
  assert.deepEqual(Array.from({length: 3}, () => a.uint32()), extra);
});

test('uint32 sequence matches xorshift32 reference; next() in [0,1)', () => {
  const rng = createRng(0x9e3779b9);
  let ref = 0x9e3779b9n;
  const advance = (x) => {
    x = (x ^ ((x << 13n) & MASK)) & MASK;
    x = x ^ (x >> 17n);
    x = (x ^ ((x << 5n) & MASK)) & MASK;
    return x;
  };
  for (let i = 0; i < 10; i += 1) {
    ref = advance(ref);
    const v = rng.uint32();
    assert.equal(v, Number(ref));
    assert.equal(rng.snapshot(), Number(ref));
    ref = advance(ref);
    const n = rng.next();
    assert.ok(n >= 0 && n < 1);
    assert.equal(rng.snapshot(), Number(ref));
    assert.equal(rng.snapshot() / 4294967296, n);
  }
});

test('rejects invalid seeds, invalid restores keep state; hashSeed matches reference', () => {
  for (const bad of ['1', -1, 0x100000000, 1.5, null, undefined, NaN, Infinity, 42n]) {
    assert.throws(() => createRng(bad), Error);
  }
  const rng = createRng(7);
  const before = rng.snapshot();
  for (const bad of [0, -1, 0x100000000, 1.5, '7', null, NaN]) {
    assert.throws(() => rng.restore(bad), Error);
  }
  assert.equal(rng.snapshot(), before);

  assert.equal(hashSeed(''), 2166136261);
  for (const text of ['abc', 'flap-12345', 'grüß 👍']) {
    assert.equal(hashSeed(text), refHash(text));
  }
  assert.throws(() => hashSeed(123), Error);
});

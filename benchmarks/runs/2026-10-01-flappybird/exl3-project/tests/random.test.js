import test from 'node:test';
import assert from 'node:assert/strict';
import {createRng, hashSeed} from '../src/random.js';

test('rng reproducibility and zero-seed initialization', () => {
  const a = createRng(0);
  const b = createRng(0);
  assert.equal(a.uint32(), b.uint32());
  assert.notEqual(a.uint32(), 0);
  const c = createRng(12345);
  const n = c.next();
  assert.ok(n >= 0 && n < 1);
  assert.throws(() => createRng(4294967296));
  assert.throws(() => createRng(-1));
  assert.throws(() => createRng('12345'));
});

test('rng snapshot and restore semantics', () => {
  const r = createRng(7);
  r.uint32();
  const snap = r.snapshot();
  assert.ok(Number.isInteger(snap) && snap >= 0 && snap <= 0xffffffff);
  r.restore(snap);
  assert.equal(r.snapshot(), snap);
  const before = r.snapshot();
  assert.throws(() => r.restore(0));
  assert.throws(() => r.restore(NaN));
  assert.throws(() => r.restore('5'));
  assert.equal(r.snapshot(), before);
  r.restore(5);
  assert.equal(r.snapshot(), 5);
});

test('hashSeed FNV-1a over UTF-16 code units', () => {
  assert.equal(hashSeed(''), 2166136261);
  const s = 'abc';
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  assert.equal(hashSeed(s), h >>> 0);
  // lone surrogate is a single code unit
  let h2 = 2166136261;
  h2 = Math.imul(h2 ^ 0xd800, 16777619);
  assert.equal(hashSeed('\ud800'), h2 >>> 0);
});

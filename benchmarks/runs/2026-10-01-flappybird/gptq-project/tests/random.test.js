import test from 'node:test';
import assert from 'node:assert/strict';
import {createRng, hashSeed} from '../src/random.js';

test('zero seed uses fixed constant; sequences deterministic; separate instances', () => {
  const a = createRng(0);
  const b = createRng(0);
  const first = a.uint32();
  assert.ok(Number.isInteger(first) && first >= 1 && first <= 0xffffffff);
  assert.equal(first, 1085196063);
  assert.equal(first, b.uint32());
  const n = a.next();
  assert.ok(n >= 0 && n < 1);
  assert.equal(n, b.next());
  assert.throws(() => createRng(-1));
  assert.throws(() => createRng(1.5));
  assert.throws(() => createRng('1'));
  assert.throws(() => createRng(0x100000000));
});

test('snapshot returns current state; restore replaces it; invalid restore throws without changing state', () => {
  const r = createRng(7);
  r.uint32();
  const snap = r.snapshot();
  assert.equal(snap, 1892583);
  assert.throws(() => r.restore(0));
  assert.throws(() => r.restore('x'));
  assert.throws(() => r.restore(0x100000000));
  assert.equal(r.snapshot(), snap, 'failed restores must not change state');
  const v2 = r.uint32();
  assert.equal(v2, 470389255);
  r.restore(snap);
  assert.equal(r.restore(snap), undefined);
  assert.equal(r.snapshot(), snap);
  assert.equal(r.uint32(), v2, 'restored state reproduces the following value');
});

test('hashSeed is FNV-1a over UTF-16 code units, returned unsigned', () => {
  assert.equal(hashSeed(''), 2166136261);
  assert.equal(hashSeed('a'), 3826002220);
  const surrogates = hashSeed('\u{1F680}');
  assert.ok(surrogates >= 0 && surrogates <= 0xffffffff);
  assert.notEqual(surrogates, hashSeed('a'));
});

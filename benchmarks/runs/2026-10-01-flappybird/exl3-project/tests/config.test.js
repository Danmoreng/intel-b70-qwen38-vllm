import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULT_CONFIG, validateConfig, serializeConfig, parseConfig} from '../src/config.js';

test('validateConfig defaults, immutability and joint constraints', () => {
  const full = validateConfig({});
  assert.deepEqual(full, DEFAULT_CONFIG);
  assert.ok(Object.isFrozen(DEFAULT_CONFIG));
  const input = {gravity: 1.5};
  const out = validateConfig(input);
  assert.equal(out.gravity, 1.5);
  assert.equal(out.width, 480);
  assert.equal(input.gravity, 1.5);
  assert.throws(() => validateConfig({width: 400, groundHeight: 250}));
  assert.throws(() => validateConfig({height: 300, groundHeight: 100, gapHeight: 200}));
  assert.throws(() => validateConfig({width: 240, birdX: 200, birdRadius: 50}));
});

test('validateConfig rejects bad values and unknown/inherited keys', () => {
  assert.throws(() => validateConfig(null));
  assert.throws(() => validateConfig([1, 2]));
  assert.throws(() => validateConfig({width: 199}));
  assert.throws(() => validateConfig({width: NaN}));
  assert.throws(() => validateConfig({width: Infinity}));
  assert.throws(() => validateConfig({seed: -1}));
  assert.throws(() => validateConfig({seed: 0xffffffff + 1}));
  assert.throws(() => validateConfig({gapHeight: 59}));
  assert.throws(() => validateConfig({gapHeight: 60.5}));
  assert.throws(() => validateConfig({extra: 1}));
  const sub = Object.create({gravity: 9});
  assert.equal(validateConfig(sub).gravity, 0.35);
});

test('serializeConfig key order and parseConfig round trip', () => {
  const s = serializeConfig({seed: 42, width: 500});
  const obj = JSON.parse(s);
  assert.deepEqual(Object.keys(obj), Object.keys(DEFAULT_CONFIG));
  assert.equal(obj.seed, 42);
  assert.equal(obj.width, 500);
  assert.ok(!s.includes(' '));
  assert.deepEqual(parseConfig(s), obj);
  assert.throws(() => parseConfig('{oops'));
  assert.throws(() => parseConfig('{"bogus":1}'));
});

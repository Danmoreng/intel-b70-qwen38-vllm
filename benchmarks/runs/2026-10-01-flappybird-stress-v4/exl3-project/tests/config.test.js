import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_CONFIG,
  validateConfig,
  serializeConfig,
  parseConfig
} from '../src/config.js';

test('DEFAULT_CONFIG is frozen and validate fills defaults without mutation', () => {
  assert.ok(Object.isFrozen(DEFAULT_CONFIG));
  const input = {gravity: 0.5};
  const result = validateConfig(input);
  assert.deepEqual(result, {...DEFAULT_CONFIG, gravity: 0.5});
  assert.deepEqual(input, {gravity: 0.5});
});

test('inherited and unknown keys are handled', () => {
  const subclass = Object.create({gravity: 1});
  const result = validateConfig(subclass);
  assert.equal(result.gravity, DEFAULT_CONFIG.gravity);
  assert.throws(() => validateConfig({bogus: 1}));
});

test('joint constraints and serialize/parse roundtrip', () => {
  assert.throws(() => validateConfig({gapHeight: 720}));
  assert.throws(() => validateConfig({height: 200, groundHeight: 100}));
  assert.throws(() => validateConfig({birdX: 5}));
  const custom = {height: 800, seed: 0};
  assert.deepEqual(JSON.parse(serializeConfig(custom)), validateConfig(custom));
  assert.equal(parseConfig(serializeConfig(custom)).seed, 0);
});

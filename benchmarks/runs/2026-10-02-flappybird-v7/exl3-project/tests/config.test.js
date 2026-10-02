import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_CONFIG,
  validateConfig,
  serializeConfig,
  parseConfig
} from '../src/config.js';

test('DEFAULT_CONFIG is deep frozen and ordered', () => {
  assert.ok(Object.isFrozen(DEFAULT_CONFIG));
  assert.deepEqual(Object.keys(DEFAULT_CONFIG), [
    'width', 'height', 'birdX', 'birdRadius', 'gravity', 'flapVelocity',
    'pipeSpeed', 'pipeWidth', 'gapHeight', 'spawnEvery', 'groundHeight', 'seed'
  ]);
  assert.throws(() => { DEFAULT_CONFIG.width = 1; });
});

test('inherited properties ignored; input not mutated', () => {
  const base = Object.create({seed: 999, bogus: 1});
  const input = {width: 400};
  Object.setPrototypeOf(input, base);
  const out = validateConfig(input);
  assert.equal(out.width, 400);
  assert.equal(out.seed, 12345);
  assert.equal(Object.hasOwn(input, 'seed'), false);
  assert.deepEqual(input, {width: 400});
  assert.notEqual(out, input);
});

test('joint constraints and parse errors', () => {
  assert.throws(() => validateConfig({height: 200, groundHeight: 0, gapHeight: 161}));
  assert.doesNotThrow(() => validateConfig({height: 300, gapHeight: 300 - 60 - 40}));
  assert.throws(() => parseConfig('nope'), SyntaxError);
  assert.throws(() => parseConfig('"str"'));
  assert.equal(parseConfig('{}').seed, 12345);
  assert.equal(
    serializeConfig({seed: 7}),
    JSON.stringify(DEFAULT_CONFIG).replace('"seed":12345', '"seed":7')
  );
});

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULT_CONFIG, validateConfig, serializeConfig, parseConfig} from '../src/config.js';

const D = DEFAULT_CONFIG;

test('defaults, deep freeze, subset filling and no mutation', () => {
  assert.deepEqual(Object.keys(D), [
    'width', 'height', 'birdX', 'birdRadius', 'gravity', 'flapVelocity',
    'pipeSpeed', 'pipeWidth', 'gapHeight', 'spawnEvery', 'groundHeight', 'seed',
  ]);
  assert.ok(Object.isFrozen(D), 'DEFAULT_CONFIG is frozen');
  for (const v of Object.values(D)) {
    if (typeof v === 'object' && v !== null) {
      assert.ok(Object.isFrozen(v), 'nested values are frozen');
    }
  }

  assert.deepEqual(validateConfig({}), D);
  const partial = {gapHeight: 200, seed: 0};
  const filled = validateConfig(partial);
  assert.equal(filled.gapHeight, 200);
  assert.equal(filled.seed, 0);
  assert.equal(filled.width, D.width);
  assert.notEqual(filled, D);
  Object.freeze(partial);
  validateConfig(partial);
  assert.deepEqual(partial, {gapHeight: 200, seed: 0}, 'input not mutated');

  // Inherited properties are ignored, not consumed.
  const parent = {width: 1000};
  const child = Object.create(parent);
  const result = validateConfig(child);
  assert.equal(result.width, D.width);
  assert.deepEqual(result, D);
});

test('serializeConfig is compact, in DEFAULT_CONFIG order; parseConfig round trips', () => {
  const cfg = validateConfig({width: 640, seed: 9});
  const text = serializeConfig(cfg);
  assert.equal(text, `{"width":640,"height":720,"birdX":120,"birdRadius":12,"gravity":0.35,`
    + `"flapVelocity":-6,"pipeSpeed":2,"pipeWidth":64,"gapHeight":180,"spawnEvery":100,`
    + `"groundHeight":60,"seed":9}`);
  const parsed = parseConfig(text);
  assert.deepEqual(parsed, cfg);
  assert.throws(() => parseConfig('{"width":'), SyntaxError);
  assert.throws(() => parseConfig('42'), TypeError);
  assert.throws(() => serializeConfig({nope: 1}), Error);
  assert.throws(() => parseConfig('["x"]'), Error);
});

test('per-key and joint range constraints are enforced', () => {
  for (const key of ['width', 'height']) {
    assert.throws(() => validateConfig({[key]: 199}), Error);
    assert.throws(() => validateConfig({[key]: 2049}), Error);
    assert.throws(() => validateConfig({[key]: 240.5}), Error);
  }
  assert.throws(() => validateConfig({gravity: 0}), Error);
  assert.throws(() => validateConfig({gravity: 2.5}), Error);
  assert.throws(() => validateConfig({flapVelocity: 0}), Error);
  assert.throws(() => validateConfig({flapVelocity: -20.5}), Error);
  assert.throws(() => validateConfig({pipeWidth: 9.5}), Error);
  assert.throws(() => validateConfig({seed: 0x100000000}), Error);
  assert.equal(validateConfig({seed: 0x100000000 - 1}).seed, 0xffffffff);
  for (const bad of [null, [], NaN, Infinity, true, '1']) {
    assert.throws(() => validateConfig({gravity: bad}), Error);
  }
  assert.throws(() => validateConfig({width: null}), Error);
  assert.throws(() => validateConfig({width: [480]}), Error);

  // Joint constraints after defaults.
  assert.throws(() => validateConfig({birdX: 12, birdRadius: 12}), Error);
  assert.throws(() => validateConfig({birdX: 468, birdRadius: 12}), Error);
  assert.ok(validateConfig({birdX: 13}));
  assert.ok(validateConfig({birdX: 467}));

  // gapHeight max = height - groundHeight - 40.
  assert.throws(() => validateConfig({gapHeight: 621}), Error);
  assert.throws(() => validateConfig({height: 400, groundHeight: 200}), Error);
  assert.ok(validateConfig({gapHeight: 620}));
  assert.throws(() => validateConfig({height: 200, groundHeight: 99}), Error);
  assert.throws(() => validateConfig({height: 200, groundHeight: 98}), Error);
  assert.ok(validateConfig({height: 200, groundHeight: 98, gapHeight: 60}));
});

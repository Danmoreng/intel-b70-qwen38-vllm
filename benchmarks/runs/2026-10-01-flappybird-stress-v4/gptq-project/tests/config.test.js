import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULT_CONFIG, validateConfig, serializeConfig, parseConfig} from '../src/config.js';

test('defaults, subset merge, inherited properties ignored, input not mutated', () => {
  assert.deepEqual(validateConfig({}), DEFAULT_CONFIG);
  assert.ok(Object.isFrozen(DEFAULT_CONFIG), 'DEFAULT_CONFIG is deep frozen');
  const got = validateConfig({width: 512, seed: 1});
  assert.equal(got.height, DEFAULT_CONFIG.height);
  assert.equal(got.width, 512);
  const proto = {bogusKey: 1, gravity: 9.9};
  const merged = validateConfig(Object.assign(Object.create(proto), {width: 500}));
  assert.equal(merged.width, 500);
  assert.equal(merged.gravity, DEFAULT_CONFIG.gravity);
  assert.ok(!('bogusKey' in merged));
  const input = {width: 500};
  validateConfig(input);
  assert.deepEqual(input, {width: 500}, 'input is never mutated');
});

test('invalid structures and out-of-range values are rejected', () => {
  for (const bad of [
    null, 42, 'x', [1], {width: 199}, {width: 2049}, {width: 480.5},
    {seed: 0, seed2: 0}, // seed2 is an unknown key
    {groundHeight: 620}, // height-groundHeight = 100, not >100
    {gapHeight: 621}, // max is 720-60-40 = 620
    {gapHeight: 59.5}, {spawnEvery: 9}, {spawnEvery: 1001},
    {pipeWidth: 9}, {pipeWidth: 201}, {pipeWidth: 64.5},
    {gravity: 0.005}, {gravity: 2.5}, {flapVelocity: 0}, {flapVelocity: -20.5},
    {pipeSpeed: 0.05}, {pipeSpeed: 10.5},
    {birdRadius: 3.5}, {birdRadius: 40.5},
    {birdX: 12}, // must be > birdRadius(12)
    {birdX: 468}, // must be < width(480)-birdRadius(12)
    {seed: -1}, {seed: 4294967296}, {seed: 12.5},
    {width: NaN}, {width: Infinity}, {height: -Infinity}, {width: null}, {width: {}},
    {width: 480, surprise: 1},
  ]) {
    assert.throws(() => validateConfig(bad), undefined, `rejects ${JSON.stringify(bad, null)} `);
  }
  // boundary values that must be accepted
  assert.doesNotThrow(() => validateConfig({width: 200, height: 200, gapHeight: 61, birdX: 50, groundHeight: 99}));
  assert.doesNotThrow(() => validateConfig({seed: 0, width: 2048}));
});

test('serialize is compact in DEFAULT_CONFIG key order; parse round-trips', () => {
  const json = serializeConfig(DEFAULT_CONFIG);
  assert.equal(
    json,
    '{"width":480,"height":720,"birdX":120,"birdRadius":12,"gravity":0.35,"flapVelocity":-6,' +
    '"pipeSpeed":2,"pipeWidth":64,"gapHeight":180,"spawnEvery":100,"groundHeight":60,"seed":12345}'
  );
  assert.deepEqual(parseConfig(json), DEFAULT_CONFIG);
  assert.deepEqual(parseConfig(serializeConfig({seed: 7})).seed, 7);
  assert.throws(() => parseConfig('{not json'));
  assert.throws(() => parseConfig('{"width":1}'));
});

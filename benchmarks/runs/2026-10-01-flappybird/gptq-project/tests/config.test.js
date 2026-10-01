import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULT_CONFIG, validateConfig, serializeConfig, parseConfig} from '../src/config.js';

test('defaults are deep frozen, subsets complete with input untouched, unknown keys rejected', () => {
  assert.ok(Object.isFrozen(DEFAULT_CONFIG));
  const src = {width: 300, extra: 1};
  assert.throws(() => validateConfig(src));
  const src2 = {width: 300};
  const cfg = validateConfig(src2);
  assert.deepEqual(cfg, {...DEFAULT_CONFIG, width: 300});
  assert.deepEqual(src2, {width: 300});
  assert.throws(() => validateConfig(null));
  assert.throws(() => validateConfig([1, 2]));
});

test('inherited properties ignored; type, range, NaN/infinity and joint constraints enforced', () => {
  class Base {}
  Base.prototype.seed = 999;
  Base.prototype.width = 99999;
  const cfg = validateConfig(new Base());
  assert.equal(cfg.seed, 12345);
  assert.equal(cfg.width, 480);
  assert.throws(() => validateConfig({gravity: 5}));
  assert.throws(() => validateConfig({width: 100}));
  assert.throws(() => validateConfig({flapVelocity: 6}));
  assert.throws(() => validateConfig({gapHeight: 700}));
  assert.throws(() => validateConfig({spawnEvery: 10.5}));
  assert.throws(() => validateConfig({seed: NaN}));
  assert.throws(() => validateConfig({gravity: Infinity}));
  assert.throws(() => validateConfig({birdX: 12, birdRadius: 12}));
  assert.throws(() => validateConfig({height: 720, groundHeight: 620}));
  assert.throws(() => validateConfig({width: 300, birdX: 290, birdRadius: 12}));
  const ok = validateConfig({width: 400, gapHeight: 480, groundHeight: 200});
  assert.equal(ok.gapHeight, 480);
});

test('serialize emits compact JSON in default order; parse round-trips; invalid text throws', () => {
  const src = {gravity: 0.5, width: 300};
  assert.equal(
    serializeConfig(src),
    '{"width":300,"height":720,"birdX":120,"birdRadius":12,"gravity":0.5,"flapVelocity":-6,"pipeSpeed":2,"pipeWidth":64,"gapHeight":180,"spawnEvery":100,"groundHeight":60,"seed":12345}'
  );
  const cfg = parseConfig('{"seed":5,"width":256}');
  assert.equal(cfg.seed, 5);
  assert.equal(cfg.height, 720);
  assert.throws(() => parseConfig('{nope'));
  assert.throws(() => parseConfig('{"bogus":1}'));
});

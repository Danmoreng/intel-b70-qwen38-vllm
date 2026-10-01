// Implementation of specs/config.md — validated, never-mutated game configuration.
// Pure module: no DOM, no wall clock. Safe to import in Node.

export const DEFAULT_CONFIG = Object.freeze({
  width: 480,
  height: 720,
  birdX: 120,
  birdRadius: 12,
  gravity: 0.35,
  flapVelocity: -6,
  pipeSpeed: 2,
  pipeWidth: 64,
  gapHeight: 180,
  spawnEvery: 100,
  groundHeight: 60,
  seed: 12345,
});

const KEYS = Object.freeze(Object.keys(DEFAULT_CONFIG));

const isFinite = (v) => typeof v === 'number' && Number.isFinite(v);
const isInt = (v) => typeof v === 'number' && Number.isInteger(v);
const inRange = (v, lo, hi) => v >= lo && v <= hi;

function fail(range, message) {
  throw range ? new RangeError(message) : new TypeError(message);
}

export function validateConfig(value) {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    fail(false, 'config must be a plain object');
  }

  const out = {};
  for (const key of KEYS) {
    out[key] = DEFAULT_CONFIG[key];
  }

  for (const key of Object.getOwnPropertyNames(value)) {
    if (!Object.hasOwn(value, key)) continue; // ignore inherited properties
    if (!Object.prototype.hasOwnProperty.call(DEFAULT_CONFIG, key)) {
      fail(false, `unknown config key: ${key}`);
    }
    const v = value[key];
    if (v === null || typeof v === 'object') {
      fail(false, `config key ${key} must be a number`);
    }
    if (!isFinite(v)) {
      fail(false, `config key ${key} must be finite (no NaN or infinity)`);
    }
    out[key] = v;
  }
  for (const sym of Object.getOwnPropertySymbols(value)) {
    fail(false, 'unknown config key');
  }

  // Joint constraints are checked after defaults are applied.
  const {
    width, height, birdX, birdRadius, gravity, flapVelocity,
    pipeSpeed, pipeWidth, gapHeight, spawnEvery, groundHeight, seed,
  } = out;

  if (!isInt(width) || !inRange(width, 200, 2048)) {
    fail(true, 'width must be a positive integer 200..2048');
  }
  if (!isInt(height) || !inRange(height, 200, 2048)) {
    fail(true, 'height must be a positive integer 200..2048');
  }
  if (!isFinite(birdRadius) || !inRange(birdRadius, 4, 40)) {
    fail(true, 'birdRadius must be finite 4..40');
  }
  if (!isFinite(gravity) || !inRange(gravity, 0.01, 2)) {
    fail(true, 'gravity must be finite 0.01..2');
  }
  if (!isFinite(flapVelocity) || flapVelocity < -20 || flapVelocity > -0.1) {
    fail(true, 'flapVelocity must be finite -20..-0.1');
  }
  if (!isFinite(pipeSpeed) || !inRange(pipeSpeed, 0.1, 10)) {
    fail(true, 'pipeSpeed must be finite 0.1..10');
  }
  if (!isInt(pipeWidth) || !inRange(pipeWidth, 10, 200)) {
    fail(true, 'pipeWidth must be an integer 10..200');
  }
  if (!isInt(gapHeight) || gapHeight < 60 || gapHeight > height - groundHeight - 40) {
    fail(true, 'gapHeight must be an integer 60..height-groundHeight-40');
  }
  if (!isInt(spawnEvery) || !inRange(spawnEvery, 10, 1000)) {
    fail(true, 'spawnEvery must be an integer 10..1000');
  }
  if (!isInt(groundHeight) || !inRange(groundHeight, 0, 200) || height - groundHeight <= 100) {
    fail(true, 'groundHeight must be an integer 0..200 with height-groundHeight>100');
  }
  if (!isFinite(birdX) || birdX <= birdRadius || birdX >= width - birdRadius) {
    fail(true, 'birdX must be finite with birdRadius<birdX<width-birdRadius');
  }
  if (!isInt(seed) || seed < 0 || seed > 0xffffffff) {
    fail(true, 'seed must be an unsigned 32-bit integer');
  }

  return out;
}

export function serializeConfig(value) {
  // JSON.stringify emits keys in insertion order (DEFAULT_CONFIG order) with no spaces.
  return JSON.stringify(validateConfig(value));
}

export function parseConfig(text) {
  return validateConfig(JSON.parse(text));
}

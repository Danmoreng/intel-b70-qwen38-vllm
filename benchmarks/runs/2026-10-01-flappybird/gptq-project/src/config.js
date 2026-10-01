// Public API contract: specs/config.md

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
  seed: 12345
});

const KEYS = Object.keys(DEFAULT_CONFIG);
const MAX_UINT32 = 0xffffffff;

function isPlainObject(v) {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

// Shared type gate: configuration values are numbers only; rejects
// null, arrays and non-number values (as well as NaN/infinity handled per key).
function checkNumber(key, v) {
  if (typeof v !== 'number' || !isFinite(v)) {
    throw new TypeError(`config key "${key}" must be a finite number`);
  }
}

function checkRange(key, v, lo, hi, label) {
  if (v < lo || v > hi) {
    throw new RangeError(`config key "${key}" must be within ${label}`);
  }
}

function checkInt(key, v) {
  if (!Number.isInteger(v)) {
    throw new RangeError(`config key "${key}" must be an integer`);
  }
}

// Accepts an object with a subset of the known keys and returns a new,
// complete configuration object; never mutates the input.
export function validateConfig(value) {
  if (!isPlainObject(value)) {
    throw new TypeError('config must be a plain object');
  }
  // Only own enumerable properties are consumed; inherited ones are ignored.
  const own = Object.keys(value);
  for (const key of own) {
    if (!Object.prototype.hasOwnProperty.call(DEFAULT_CONFIG, key)) {
      throw new RangeError(`config has unknown key "${key}"`);
    }
  }

  // Start from a copy of the defaults, then overlay own properties.
  const cfg = { ...DEFAULT_CONFIG };
  for (const key of own) {
    cfg[key] = value[key];
  }

  // Per-key checks on the completed values.
  checkNumber('width', cfg.width);
  checkRange('width', cfg.width, 200, 2048, '200..2048');
  checkNumber('height', cfg.height);
  checkRange('height', cfg.height, 200, 2048, '200..2048');
  checkNumber('birdRadius', cfg.birdRadius);
  checkRange('birdRadius', cfg.birdRadius, 4, 40, '4..40');
  checkNumber('gravity', cfg.gravity);
  checkRange('gravity', cfg.gravity, 0.01, 2, '0.01..2');
  checkNumber('flapVelocity', cfg.flapVelocity);
  checkRange('flapVelocity', cfg.flapVelocity, -20, -0.1, '-20..-0.1');
  checkNumber('pipeSpeed', cfg.pipeSpeed);
  checkRange('pipeSpeed', cfg.pipeSpeed, 0.1, 10, '0.1..10');
  checkNumber('pipeWidth', cfg.pipeWidth);
  checkInt('pipeWidth', cfg.pipeWidth);
  checkRange('pipeWidth', cfg.pipeWidth, 10, 200, '10..200');
  checkNumber('gapHeight', cfg.gapHeight);
  checkInt('gapHeight', cfg.gapHeight);
  checkNumber('spawnEvery', cfg.spawnEvery);
  checkInt('spawnEvery', cfg.spawnEvery);
  checkRange('spawnEvery', cfg.spawnEvery, 10, 1000, '10..1000');
  checkNumber('groundHeight', cfg.groundHeight);
  checkInt('groundHeight', cfg.groundHeight);
  checkRange('groundHeight', cfg.groundHeight, 0, 200, '0..200');
  checkNumber('seed', cfg.seed);
  checkInt('seed', cfg.seed);
  if (cfg.seed < 0 || cfg.seed > MAX_UINT32) {
    throw new RangeError('config key "seed" must be an unsigned 32-bit integer');
  }

  // Joint constraints, evaluated after defaults are applied.
  checkNumber('birdX', cfg.birdX);
  if (!(cfg.birdX > cfg.birdRadius) || !(cfg.birdX < cfg.width - cfg.birdRadius)) {
    throw new RangeError('birdX must satisfy birdRadius < birdX < width - birdRadius');
  }
  const gapMax = cfg.height - cfg.groundHeight - 40;
  if (cfg.gapHeight < 60 || cfg.gapHeight > gapMax) {
    throw new RangeError(`config key "gapHeight" must be within 60..${gapMax}`);
  }
  if (!(cfg.height - cfg.groundHeight > 100)) {
    throw new RangeError('height - groundHeight must be greater than 100');
  }

  return cfg;
}

// Compact JSON with keys in DEFAULT_CONFIG order.
export function serializeConfig(value) {
  const cfg = validateConfig(value);
  const ordered = {};
  for (const key of KEYS) {
    ordered[key] = cfg[key];
  }
  return JSON.stringify(ordered);
}

// Parse JSON text, then validate the parsed object.
export function parseConfig(text) {
  return validateConfig(JSON.parse(text));
}

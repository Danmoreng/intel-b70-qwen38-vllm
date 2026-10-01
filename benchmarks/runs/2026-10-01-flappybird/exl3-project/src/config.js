// Game configuration API per specs/config.md.

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

const KEYS = Object.keys(DEFAULT_CONFIG);
const MAX_UINT32 = 0xffffffff;

function isInt(v) {
  return typeof v === 'number' && Number.isInteger(v);
}

function isFiniteNumber(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function checkOne(key, value) {
  switch (key) {
    case 'width':
    case 'height':
      if (!isInt(value) || value < 200 || value > 2048) {
        throw new RangeError(`${key} must be an integer between 200 and 2048`);
      }
      return;
    case 'birdRadius':
      if (!isFiniteNumber(value) || value < 4 || value > 40) {
        throw new RangeError('birdRadius must be finite between 4 and 40');
      }
      return;
    case 'gravity':
      if (!isFiniteNumber(value) || value < 0.01 || value > 2) {
        throw new RangeError('gravity must be finite between 0.01 and 2');
      }
      return;
    case 'flapVelocity':
      if (!isFiniteNumber(value) || value < -20 || value > -0.1) {
        throw new RangeError('flapVelocity must be finite between -20 and -0.1');
      }
      return;
    case 'pipeSpeed':
      if (!isFiniteNumber(value) || value < 0.1 || value > 10) {
        throw new RangeError('pipeSpeed must be finite between 0.1 and 10');
      }
      return;
    case 'pipeWidth':
      if (!isInt(value) || value < 10 || value > 200) {
        throw new RangeError('pipeWidth must be an integer between 10 and 200');
      }
      return;
    case 'gapHeight':
      if (!isInt(value) || value < 60) {
        throw new RangeError('gapHeight must be an integer at least 60');
      }
      return;
    case 'spawnEvery':
      if (!isInt(value) || value < 10 || value > 1000) {
        throw new RangeError('spawnEvery must be an integer between 10 and 1000');
      }
      return;
    case 'groundHeight':
      if (!isInt(value) || value < 0 || value > 200) {
        throw new RangeError('groundHeight must be an integer between 0 and 200');
      }
      return;
    case 'seed':
      if (!isInt(value) || value < 0 || value > MAX_UINT32) {
        throw new RangeError('seed must be an unsigned 32-bit integer');
      }
      return;
    case 'birdX':
      if (!isFiniteNumber(value)) {
        throw new RangeError('birdX must be a finite number');
      }
      return;
    default:
      throw new TypeError(`unknown configuration key: ${key}`);
  }
}

export function validateConfig(value) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError('configuration must be a non-null, non-array object');
  }
  const own = Object.keys(value);
  for (const key of own) {
    if (!KEYS.includes(key)) {
      throw new TypeError(`unknown configuration key: ${key}`);
    }
  }
  const result = {};
  for (const key of KEYS) {
    if (own.includes(key)) {
      const v = value[key];
      if (v === null || Array.isArray(v) || typeof v !== 'number' || !Number.isFinite(v)) {
        throw new TypeError(`configuration value for ${key} must be a finite number`);
      }
      checkOne(key, v);
      result[key] = v;
    } else {
      result[key] = DEFAULT_CONFIG[key];
    }
  }
  if (!(result.birdX > result.birdRadius && result.birdX < result.width - result.birdRadius)) {
    throw new RangeError('birdX must be greater than birdRadius and less than width - birdRadius');
  }
  if (!(result.height - result.groundHeight > 100)) {
    throw new RangeError('height - groundHeight must be greater than 100');
  }
  if (!(result.gapHeight <= result.height - result.groundHeight - 40)) {
    throw new RangeError('gapHeight must not exceed height - groundHeight - 40');
  }
  return result;
}

export function serializeConfig(value) {
  const config = validateConfig(value);
  const ordered = {};
  for (const key of KEYS) {
    ordered[key] = config[key];
  }
  return JSON.stringify(ordered);
}

export function parseConfig(text) {
  return validateConfig(JSON.parse(text));
}

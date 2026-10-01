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

const KNOWN_KEYS = Object.freeze(Object.keys(DEFAULT_CONFIG));

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function isIntegerInRange(value, min, max) {
  return Number.isInteger(value) && value >= min && value <= max;
}

function checkFinite(key, value) {
  if (!isFiniteNumber(value)) {
    throw new TypeError(`${key} must be a finite number`);
  }
}

function checkRange(key, value, min, max) {
  if (!isFiniteNumber(value) || value < min || value > max) {
    throw new RangeError(`${key} must be in the range ${min}..${max}`);
  }
}

function checkInteger(key, value, min, max) {
  if (!isIntegerInRange(value, min, max)) {
    throw new RangeError(`${key} must be an integer in the range ${min}..${max}`);
  }
}

function checkUint32(key, value) {
  if (!isIntegerInRange(value, 0, 0xffffffff)) {
    throw new RangeError(`${key} must be an unsigned 32-bit integer`);
  }
}

export function validateConfig(value) {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new TypeError('config must be an object');
  }

  const provided = {};
  for (const key of Object.keys(value)) {
    if (!KNOWN_KEYS.includes(key)) {
      throw new TypeError(`unknown config key: ${key}`);
    }
    provided[key] = value[key];
  }

  const merged = {};
  for (const key of KNOWN_KEYS) {
    merged[key] = key in provided ? provided[key] : DEFAULT_CONFIG[key];
  }

  checkInteger('width', merged.width, 200, 2048);
  checkInteger('height', merged.height, 200, 2048);
  checkFinite('birdX', merged.birdX);
  checkRange('birdRadius', merged.birdRadius, 4, 40);
  checkRange('gravity', merged.gravity, 0.01, 2);
  checkRange('flapVelocity', merged.flapVelocity, -20, -0.1);
  checkRange('pipeSpeed', merged.pipeSpeed, 0.1, 10);
  checkInteger('pipeWidth', merged.pipeWidth, 10, 200);
  checkInteger('gapHeight', merged.gapHeight, 60, Number.POSITIVE_INFINITY);
  checkInteger('spawnEvery', merged.spawnEvery, 10, 1000);
  checkInteger('groundHeight', merged.groundHeight, 0, 200);
  checkUint32('seed', merged.seed);

  if (merged.height - merged.groundHeight <= 100) {
    throw new RangeError('height - groundHeight must be greater than 100');
  }
  if (merged.gapHeight > merged.height - merged.groundHeight - 40) {
    throw new RangeError(
      'gapHeight must not exceed height - groundHeight - 40'
    );
  }
  if (merged.birdX <= merged.birdRadius) {
    throw new RangeError('birdX must be greater than birdRadius');
  }
  if (merged.birdX >= merged.width - merged.birdRadius) {
    throw new RangeError('birdX must be less than width - birdRadius');
  }

  return merged;
}

export function serializeConfig(value) {
  const config = validateConfig(value);
  const ordered = {};
  for (const key of KNOWN_KEYS) {
    ordered[key] = config[key];
  }
  return JSON.stringify(ordered);
}

export function parseConfig(text) {
  return validateConfig(JSON.parse(text));
}

// Public API contract: specs/config.md

const DEFAULTS = {
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
};

const KEY_ORDER = Object.keys(DEFAULTS);
const MAX_UINT32 = 0xFFFFFFFF;

function freezeDeep(value) {
  Object.freeze(value);
  for (const key of Object.keys(value)) {
    const child = value[key];
    if (child !== null && typeof child === 'object' && !Object.isFrozen(child)) {
      freezeDeep(child);
    }
  }
  return value;
}

export const DEFAULT_CONFIG = freezeDeep(Object.assign({}, DEFAULTS));

function isInteger(value) {
  return typeof value === 'number' && Number.isInteger(value);
}

function isFiniteInRange(value, min, max) {
  return (
    typeof value === 'number' &&
    Number.isFinite(value) &&
    value >= min &&
    value <= max
  );
}

function isUint32(value) {
  return isInteger(value) && value >= 0 && value <= MAX_UINT32;
}

function validateField(key, value) {
  switch (key) {
    case 'width':
    case 'height':
      if (!isInteger(value) || value < 200 || value > 2048) {
        throw new RangeError(`${key} must be an integer between 200 and 2048`);
      }
      return;
    case 'birdRadius':
      if (!isFiniteInRange(value, 4, 40)) {
        throw new RangeError('birdRadius must be a finite number between 4 and 40');
      }
      return;
    case 'gravity':
      if (!isFiniteInRange(value, 0.01, 2)) {
        throw new RangeError('gravity must be a finite number between 0.01 and 2');
      }
      return;
    case 'flapVelocity':
      if (!isFiniteInRange(value, -20, -0.1)) {
        throw new RangeError('flapVelocity must be a finite number between -20 and -0.1');
      }
      return;
    case 'pipeSpeed':
      if (!isFiniteInRange(value, 0.1, 10)) {
        throw new RangeError('pipeSpeed must be a finite number between 0.1 and 10');
      }
      return;
    case 'pipeWidth':
      if (!isInteger(value) || value < 10 || value > 200) {
        throw new RangeError('pipeWidth must be an integer between 10 and 200');
      }
      return;
    case 'spawnEvery':
      if (!isInteger(value) || value < 10 || value > 1000) {
        throw new RangeError('spawnEvery must be an integer between 10 and 1000');
      }
      return;
    case 'groundHeight':
      if (!isInteger(value) || value < 0 || value > 200) {
        throw new RangeError('groundHeight must be an integer between 0 and 200');
      }
      return;
    case 'seed':
      if (!isUint32(value)) {
        throw new RangeError('seed must be an unsigned 32-bit integer');
      }
      return;
  }
}

export function validateConfig(value) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError('validateConfig expects a non-array object');
  }
  const result = Object.assign({}, DEFAULTS);
  // Only own properties; inherited configuration values are ignored.
  for (const key of Object.keys(value)) {
    if (!Object.prototype.hasOwnProperty.call(DEFAULTS, key)) {
      throw new Error(`unknown configuration key: ${key}`);
    }
    const v = value[key];
    if (v === null || Array.isArray(v) || typeof v === 'boolean') {
      throw new TypeError(`${key} must be a number`);
    }
    validateField(key, v);
    result[key] = v;
  }
  // Joint constraints, checked after defaults are applied.
  if (
    typeof result.birdX !== 'number' ||
    !Number.isFinite(result.birdX) ||
    result.birdX <= result.birdRadius ||
    result.birdX >= result.width - result.birdRadius
  ) {
    throw new RangeError(
      'birdX must be a finite number > birdRadius and < width - birdRadius'
    );
  }
  if (
    !isInteger(result.gapHeight) ||
    result.gapHeight < 60 ||
    result.gapHeight > result.height - result.groundHeight - 40
  ) {
    throw new RangeError(
      'gapHeight must be an integer between 60 and height - groundHeight - 40'
    );
  }
  if (!(result.height - result.groundHeight > 100)) {
    throw new RangeError('height - groundHeight must be greater than 100');
  }
  return result;
}

export function serializeConfig(value) {
  const config = validateConfig(value);
  return JSON.stringify(Object.fromEntries(KEY_ORDER.map((k) => [k, config[k]])));
}

export function parseConfig(text) {
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    throw new SyntaxError('parseConfig expects JSON text: ' + err.message);
  }
  return validateConfig(parsed);
}

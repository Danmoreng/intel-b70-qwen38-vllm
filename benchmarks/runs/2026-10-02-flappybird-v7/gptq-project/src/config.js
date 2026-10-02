// Validated game configuration, per specs/config.md.
// Pure module: safe to import in Node; never mutates caller inputs.

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

const MAX_UINT32 = 0xffffffff;

function fail(message) {
  throw new TypeError(message);
}

function checkNumber(key, value, range) {
  if (value === null || typeof value !== 'number' || typeof value === 'boolean' ||
    Number.isNaN(value) || !Number.isFinite(value)) {
    fail(`config key "${key}" must be a finite number`);
  }
  const { min, max, integer } = range;
  if (integer && !Number.isInteger(value)) {
    fail(`config key "${key}" must be an integer`);
  }
  if (value < min || value > max) {
    throw new RangeError(`config key "${key}" must be between ${min} and ${max}`);
  }
}

const KEY_RULES = {
  width: { min: 200, max: 2048, integer: true },
  height: { min: 200, max: 2048, integer: true },
  birdX: { min: -Infinity, max: Infinity, integer: false },
  birdRadius: { min: 4, max: 40, integer: false },
  gravity: { min: 0.01, max: 2, integer: false },
  flapVelocity: { min: -20, max: -0.1, integer: false },
  pipeSpeed: { min: 0.1, max: 10, integer: false },
  pipeWidth: { min: 10, max: 200, integer: true },
  gapHeight: { min: 60, max: Infinity, integer: true },
  spawnEvery: { min: 10, max: 1000, integer: true },
  groundHeight: { min: 0, max: 200, integer: true },
  seed: { min: 0, max: MAX_UINT32, integer: true },
};

export function validateConfig(value) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    fail('validateConfig accepts only a plain configuration object');
  }

  const complete = { ...DEFAULT_CONFIG };
  for (const key of Object.keys(value)) {
    if (!Object.prototype.hasOwnProperty.call(KEY_RULES, key)) {
      throw new RangeError(`unknown config key "${key}"`);
    }
    const v = value[key];
    if (v === null || Array.isArray(v) ||
      typeof v !== 'number' || Number.isNaN(v) || !Number.isFinite(v)) {
      fail(`config key "${key}" must be a finite number`);
    }
    complete[key] = v;
  }

  for (const key of Object.keys(complete)) {
    checkNumber(key, complete[key], KEY_RULES[key]);
  }

  // Joint constraints, checked after defaults are applied.
  if (!(complete.birdX > complete.birdRadius &&
        complete.birdX < complete.width - complete.birdRadius)) {
    throw new RangeError('birdX must be greater than birdRadius and less than width - birdRadius');
  }
  if (complete.gapHeight > complete.height - complete.groundHeight - 40) {
    throw new RangeError('gapHeight must not exceed height - groundHeight - 40');
  }
  if (complete.height - complete.groundHeight <= 100) {
    throw new RangeError('height - groundHeight must be greater than 100');
  }

  return complete;
}

export function serializeConfig(value) {
  const complete = validateConfig(value);
  const ordered = {};
  for (const key of Object.keys(DEFAULT_CONFIG)) {
    ordered[key] = complete[key];
  }
  return JSON.stringify(ordered);
}

export function parseConfig(text) {
  return validateConfig(JSON.parse(text));
}

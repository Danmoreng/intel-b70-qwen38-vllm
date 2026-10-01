// Public API contract: specs/random.md

const MAX_UINT32 = 0xffffffff;
const ZERO_SEED_STATE = 0x6d2b79f5;

function isUint32(value) {
  return (
    typeof value === 'number' &&
    Number.isInteger(value) &&
    value >= 0 &&
    value <= MAX_UINT32
  );
}

export function createRng(seed) {
  if (!isUint32(seed)) {
    throw new TypeError('seed must be an unsigned 32-bit integer');
  }
  let state = seed === 0 ? ZERO_SEED_STATE : seed;

  function uint32() {
    let x = state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    state = x >>> 0;
    return state;
  }

  function next() {
    return uint32() / 4294967296;
  }

  function snapshot() {
    return state;
  }

  function restore(value) {
    if (!isUint32(value) || value === 0) {
      throw new TypeError('restore requires a nonzero unsigned 32-bit integer');
    }
    state = value;
    return undefined;
  }

  return { next, uint32, snapshot, restore };
}

export function hashSeed(text) {
  if (typeof text !== 'string') {
    throw new TypeError('text must be a string');
  }
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash = Math.imul(hash ^ text.charCodeAt(i), 16777619);
  }
  return hash >>> 0;
}

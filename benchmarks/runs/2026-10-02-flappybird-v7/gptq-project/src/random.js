// Seeded RNG and seed hashing, per specs/random.md.
// Pure module: safe to import in Node, no wall clock, no Math.random.

const MASK = 0xffffffff;
const ZERO_SEED_STATE = 0x6d2b79f5;
const FNV_OFFSET = 2166136261;
const FNV_PRIME = 16777619;

function requireSeed(seed) {
  if (typeof seed !== 'number' || !Number.isInteger(seed)) {
    throw new TypeError('seed must be an unsigned 32-bit integer');
  }
  if (seed < 0 || seed > MASK) {
    throw new RangeError('seed must be an unsigned 32-bit integer');
  }
  return seed;
}

export function createRng(seed) {
  seed = requireSeed(seed);
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
    return state >>> 0;
  }

  function restore(value) {
    if (typeof value !== 'number' || !Number.isInteger(value)) {
      throw new TypeError('restore accepts only a nonzero uint32');
    }
    if (value < 1 || value > MASK) {
      throw new RangeError('restore accepts only a nonzero uint32');
    }
    state = value;
  }

  return { next, uint32, snapshot, restore };
}

export function hashSeed(text) {
  if (typeof text !== 'string') {
    throw new TypeError('hashSeed accepts only strings');
  }
  let hash = FNV_OFFSET;
  for (let i = 0; i < text.length; i += 1) {
    hash = Math.imul(hash ^ text.charCodeAt(i), FNV_PRIME);
  }
  return hash >>> 0;
}

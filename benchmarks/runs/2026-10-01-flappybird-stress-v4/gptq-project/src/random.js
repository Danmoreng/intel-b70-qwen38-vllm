// Implementation of specs/random.md — seeded xorshift32 RNG and FNV-1a text hash.
// Pure module: no Date.now, no Math.random, no DOM. Safe to import in Node.

const ZERO_INIT = 0x6d2b79f5;
const MAX_UINT32 = 0xffffffff;

function isUint32(value) {
  return (
    typeof value === 'number' &&
    Number.isFinite(value) &&
    Number.isInteger(value) &&
    value >= 0 &&
    value <= MAX_UINT32
  );
}

// FNV-1a over JavaScript UTF-16 code units.
export function hashSeed(text) {
  const str = String(text);
  let hash = 2166136261;
  for (let i = 0; i < str.length; i += 1) {
    const codeUnit = str.charCodeAt(i);
    hash = Math.imul(hash ^ codeUnit, 16777619) >>> 0;
  }
  return hash >>> 0;
}

export function createRng(seed) {
  if (typeof seed !== 'number' || !Number.isFinite(seed) || !Number.isInteger(seed)) {
    throw new TypeError('seed must be an unsigned 32-bit integer');
  }
  if (seed < 0 || seed > MAX_UINT32) {
    throw new RangeError('seed must be an unsigned 32-bit integer');
  }
  let state = (seed === 0 ? ZERO_INIT : seed >>> 0) >>> 0;

  function uint32() {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    state = state >>> 0; // truncate to unsigned after the sequence
    return state;
  }

  function next() {
    return uint32() / 4294967296;
  }

  function snapshot() {
    return state >>> 0;
  }

  function restore(value) {
    // Accepts only a nonzero uint32; invalid input leaves state untouched.
    if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isInteger(value)) {
      throw new TypeError('restore accepts a nonzero uint32');
    }
    if (value <= 0 || value > MAX_UINT32) {
      throw new RangeError('restore accepts a nonzero uint32');
    }
    state = value >>> 0;
    return undefined;
  }

  return { next, uint32, snapshot, restore };
}

// Seeded random API per specs/random.md.
// No Date.now, no Math.random; independent stateful instances.

const MAX_UINT32 = 0xffffffff;

export function createRng(seed) {
  if (typeof seed !== 'number' || !Number.isInteger(seed) || seed < 0 || seed > MAX_UINT32) {
    throw new TypeError('createRng seed must be an unsigned 32-bit integer');
  }
  let x = seed === 0 ? 0x6d2b79f5 : seed >>> 0;

  function uint32() {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return x >>> 0;
  }

  return {
    next() {
      return uint32() / 4294967296;
    },
    uint32,
    snapshot() {
      return x >>> 0;
    },
    restore(value) {
      if (typeof value !== 'number' || !Number.isInteger(value) || value < 1 || value > MAX_UINT32) {
        throw new TypeError('restore state must be a nonzero unsigned 32-bit integer');
      }
      x = value >>> 0;
      return undefined;
    },
  };
}

export function hashSeed(text) {
  let hash = 2166136261;
  const s = String(text);
  for (let i = 0; i < s.length; i++) {
    hash = Math.imul(hash ^ s.charCodeAt(i), 16777619);
  }
  return hash >>> 0;
}

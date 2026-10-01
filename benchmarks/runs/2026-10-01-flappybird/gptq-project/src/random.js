// Public API contract: specs/random.md

const MAX_UINT32 = 0xffffffff;

function isUint32(v) {
  return typeof v === 'number' && Number.isInteger(v) && v >= 0 && v <= MAX_UINT32;
}

// createRng(seed): seed is an unsigned 32-bit integer, including zero; other inputs are rejected.
export function createRng(seed) {
  if (typeof seed !== 'number' || !Number.isInteger(seed)) {
    throw new TypeError('seed must be an integer');
  }
  if (seed < 0 || seed > MAX_UINT32) {
    throw new RangeError('seed must be an unsigned 32-bit integer');
  }
  // Zero seeds are initialized to the fixed xorshift32 magic constant.
  let x = seed === 0 ? 0x6d2b79f5 : seed >>> 0;

  function uint32() {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    return x;
  }

  return {
    // Deterministic only: no Date.now, no Math.random.
    next() {
      return uint32() / 4294967296;
    },
    uint32,
    snapshot() {
      return x;
    },
    // Accepts only a nonzero uint32; invalid input throws before touching state.
    restore(v) {
      if (typeof v !== 'number' || !Number.isInteger(v) || v < 1 || v > MAX_UINT32) {
        throw new RangeError('restore requires a nonzero uint32');
      }
      x = v >>> 0;
    }
  };
}

// FNV-1a over JavaScript UTF-16 code units; final value is returned unsigned.
export function hashSeed(text) {
  let h = 2166136261;
  const s = String(text);
  for (let i = 0; i < s.length; i++) {
    h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  }
  return h >>> 0;
}

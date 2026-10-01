// Public API contract: specs/input.md

const ACTIONS = ['flap', 'pause', 'restart'];

// Edge-triggered keyboard actions; no global DOM listeners.
export function createInput(bindings = {flap: ['Space', 'ArrowUp'], pause: ['KeyP'], restart: ['KeyR']}) {
  if (bindings === null || typeof bindings !== 'object' || Array.isArray(bindings)) {
    throw new TypeError('bindings must be an object');
  }
  for (const key of Object.keys(bindings)) {
    if (!ACTIONS.includes(key)) {
      throw new RangeError(`unknown binding action "${key}"`);
    }
  }
  const byCode = new Map();
  for (const action of ACTIONS) {
    const codes = bindings[action];
    if (!Array.isArray(codes)) {
      throw new TypeError(`binding "${action}" must be an array of codes`);
    }
    for (const code of codes) {
      if (typeof code !== 'string' || code === '') {
        throw new RangeError('binding codes must be nonempty strings');
      }
      if (byCode.has(code)) {
        throw new RangeError(`binding code "${code}" is not unique across actions`);
      }
      byCode.set(code, action);
    }
  }

  let held = new Set();
  let queue = [];
  let disposed = false;

  return {
    // type defaults to 'keydown'; keyup delegates to release; recognized
    // nonrepeat keydown emits its action exactly once while held.
    handle(evt) {
      if (disposed) {
        return false;
      }
      const {code, repeat, type = 'keydown'} = evt ?? {};
      const action = typeof code === 'string' ? byCode.get(code) : undefined;
      if (action === undefined) {
        return false;
      }
      if (type === 'keyup') {
        return this.release({code});
      }
      if (type !== 'keydown') {
        return false;
      }
      if (repeat === true || held.has(code)) {
        return true;
      }
      held.add(code);
      queue.push(action);
      return true;
    },
    // Recognized codes return true even when not held; removes from the held set.
    release(evt) {
      if (disposed) {
        return false;
      }
      const code = evt && evt.code;
      const recognized = typeof code === 'string' && byCode.has(code);
      if (recognized) {
        held.delete(code);
      }
      return recognized;
    },
    consume() {
      if (disposed) {
        return [];
      }
      const out = queue;
      queue = [];
      return out;
    },
    reset() {
      if (disposed) {
        return;
      }
      held = new Set();
      queue = [];
    },
    dispose() {
      disposed = true;
    }
  };
}

// Maps finite client coordinates relative to the rect into logical coordinates
// (scaled by the rect dimensions), clamped to 0..width / 0..height.
export function normalizePointer(event, rect, width, height) {
  if (event === null || typeof event !== 'object') {
    throw new TypeError('event must be an object');
  }
  const {clientX, clientY} = event;
  if (typeof clientX !== 'number' || !Number.isFinite(clientX) ||
      typeof clientY !== 'number' || !Number.isFinite(clientY)) {
    throw new RangeError('pointer clientX/clientY must be finite');
  }
  if (rect === null || typeof rect !== 'object' || Array.isArray(rect)) {
    throw new TypeError('rect must be an object');
  }
  const {left, top, width: rw, height: rh} = rect;
  for (const [name, v] of [['left', left], ['top', top], ['width', rw], ['height', rh]]) {
    if (typeof v !== 'number' || !Number.isFinite(v)) {
      throw new RangeError(`rect.${name} must be finite`);
    }
  }
  if (rw <= 0 || rh <= 0) {
    throw new RangeError('rect dimensions must be positive');
  }
  if (typeof width !== 'number' || !Number.isFinite(width) || width <= 0 ||
      typeof height !== 'number' || !Number.isFinite(height) || height <= 0) {
    throw new RangeError('logical width/height must be positive');
  }
  const x = ((clientX - left) / rw) * width;
  const y = ((clientY - top) / rh) * height;
  return {
    x: Math.min(Math.max(x, 0), width),
    y: Math.min(Math.max(y, 0), height)
  };
}

// Bounded fixed-step ticker; no requestAnimationFrame internally.
export function createTicker(step, options = {hz: 60, maxSteps: 5}) {
  if (typeof step !== 'function') {
    throw new TypeError('step must be a function');
  }
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('options must be an object');
  }
  const hz = options.hz === undefined ? 60 : options.hz;
  const maxSteps = options.maxSteps === undefined ? 5 : options.maxSteps;
  if (typeof hz !== 'number' || !Number.isFinite(hz) || hz <= 0 || hz > 240) {
    throw new RangeError('hz must be a positive finite number <= 240');
  }
  if (typeof maxSteps !== 'number' || !Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > 100) {
    throw new RangeError('maxSteps must be an integer within 1..100');
  }
  const dt = 1000 / hz;
  const dtSec = dt / 1000;

  let base = null;
  let paused = false;

  return {
    update(timestampMs) {
      if (typeof timestampMs !== 'number' || !Number.isFinite(timestampMs)) {
        throw new RangeError('timestamp must be a finite number');
      }
      if (paused) {
        return;
      }
      if (base === null) {
        base = timestampMs;
        return;
      }
      if (timestampMs < base) {
        // Atomic: no step is applied and the base is untouched.
        throw new RangeError('timestamp moved backwards');
      }
      const elapsed = timestampMs - base;
      const n = Math.floor(elapsed / dt + 1e-9);
      if (n > maxSteps) {
        for (let i = 0; i < maxSteps; i++) {
          step(dtSec);
        }
        // Drop the excess backlog; keep the fractional remainder.
        base = timestampMs - (elapsed % dt);
      } else {
        for (let i = 0; i < n; i++) {
          step(dtSec);
        }
        base += n * dt;
      }
    },
    // Prevents callbacks and clears the elapsed base.
    pause() {
      paused = true;
      base = null;
    },
    // Starts fresh: the next update re-initializes the base.
    resume() {
      paused = false;
      base = null;
    },
    reset() {
      base = null;
    }
  };
}

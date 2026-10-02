// Edge-triggered input and bounded fixed-step ticker, per specs/input.md.
// Pure module: no global DOM listeners, no wall clock, no requestAnimationFrame.

const DEFAULT_BINDINGS = {
  flap: ['Space', 'ArrowUp'],
  pause: ['KeyP'],
  restart: ['KeyR'],
};

function checkBindings(bindings) {
  if (bindings === null || typeof bindings !== 'object' || Array.isArray(bindings)) {
    throw new TypeError('bindings must be an object');
  }
  const codes = new Map();
  for (const action of Object.keys(bindings)) {
    if (!Object.prototype.hasOwnProperty.call(DEFAULT_BINDINGS, action)) {
      throw new RangeError(`unknown binding action "${action}"`);
    }
    const list = bindings[action];
    if (!Array.isArray(list)) {
      throw new TypeError(`binding "${action}" must be an array of key codes`);
    }
    for (const code of list) {
      if (typeof code !== 'string' || code === '') {
        throw new TypeError('key codes must be nonempty strings');
      }
      if (codes.has(code)) {
        throw new RangeError(`key code "${code}" is bound more than once`);
      }
      codes.set(code, action);
    }
  }
  return codes;
}

export function createInput(bindings = DEFAULT_BINDINGS) {
  const actions = checkBindings(bindings);
  const held = new Set();
  const queue = [];
  let disposed = false;

  function keydown(code) {
    const action = actions.get(code);
    if (action === undefined) {
      return false;
    }
    if (!held.has(code)) {
      held.add(code);
      queue.push(action);
    }
    return true;
  }

  function release(event) {
    if (disposed) {
      return false;
    }
    if (event === null || typeof event !== 'object' || typeof event.code !== 'string') {
      return false;
    }
    held.delete(event.code);
    return actions.has(event.code);
  }

  function handle(event) {
    if (disposed) {
      return false;
    }
    if (event === null || typeof event !== 'object' || typeof event.code !== 'string') {
      return false;
    }
    if (event.type === 'keyup') {
      return release(event);
    }
    if (event.repeat === true) {
      return actions.has(event.code);
    }
    return keydown(event.code);
  }

  function consume() {
    if (disposed) {
      return [];
    }
    return queue.splice(0, queue.length);
  }

  function reset() {
    held.clear();
    queue.length = 0;
  }

  function dispose() {
    disposed = true;
    held.clear();
    queue.length = 0;
  }

  return {handle, release, consume, reset, dispose};
}

function requireFiniteObject(obj, keys, label) {
  if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) {
    throw new TypeError(`${label} must be an object`);
  }
  for (const key of keys) {
    if (typeof obj[key] !== 'number' || !Number.isFinite(obj[key])) {
      throw new TypeError(`${label}.${key} must be a finite number`);
    }
  }
  return obj;
}

export function normalizePointer(event, rect, width, height) {
  requireFiniteObject(event, ['clientX', 'clientY'], 'event');
  requireFiniteObject(rect, ['left', 'top', 'width', 'height'], 'rect');
  if (rect.width <= 0 || rect.height <= 0) {
    throw new RangeError('rect dimensions must be positive');
  }
  if (typeof width !== 'number' || !Number.isFinite(width) || width <= 0) {
    throw new RangeError('logical width must be a positive finite number');
  }
  if (typeof height !== 'number' || !Number.isFinite(height) || height <= 0) {
    throw new RangeError('logical height must be a positive finite number');
  }
  const x = (event.clientX - rect.left) / rect.width * width;
  const y = (event.clientY - rect.top) / rect.height * height;
  return {
    x: Math.min(width, Math.max(0, x)),
    y: Math.min(height, Math.max(0, y)),
  };
}

function checkTickerOptions(options) {
  const {hz, maxSteps} = options;
  if (typeof hz !== 'number' || !Number.isFinite(hz) || hz <= 0 || hz > 240) {
    throw new RangeError('hz must be a positive finite number up to 240');
  }
  if (typeof maxSteps !== 'number' || !Number.isInteger(maxSteps) ||
    maxSteps < 1 || maxSteps > 100) {
    throw new RangeError('maxSteps must be an integer from 1 to 100');
  }
  return {hz, maxSteps};
}

export function createTicker(step, options = {hz: 60, maxSteps: 5}) {
  if (typeof step !== 'function') {
    throw new TypeError('step must be a function');
  }
  const {hz, maxSteps} = checkTickerOptions({hz: 60, maxSteps: 5, ...options});
  const dt = 1000 / hz;
  let base = null;
  let paused = false;

  function update(timestampMs) {
    if (typeof timestampMs !== 'number' || !Number.isFinite(timestampMs)) {
      throw new TypeError('update accepts only finite timestamps');
    }
    if (paused) {
      return;
    }
    if (base === null) {
      base = timestampMs;
      return;
    }
    if (timestampMs < base) {
      throw new RangeError('decreasing timestamps are rejected');
    }
    const elapsed = timestampMs - base;
    const full = Math.floor(elapsed / dt);
    const steps = Math.min(full, maxSteps);
    if (steps <= 0) {
      return;
    }
    for (let i = 0; i < steps; i += 1) {
      step(dt / 1000);
    }
    // Normal frames retain the fractional remainder; when the backlog
    // exceeds maxSteps the excess (including the remainder) is dropped.
    base = full > maxSteps ? timestampMs : base + steps * dt;
  }

  function pause() {
    paused = true;
    base = null;
  }

  function resume() {
    paused = false;
    base = null;
  }

  function reset() {
    base = null;
  }

  return {update, pause, resume, reset};
}

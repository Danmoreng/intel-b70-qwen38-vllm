// Public API contract: specs/input.md

const KNOWN_ACTIONS = ['flap', 'pause', 'restart'];
const DEFAULT_BINDINGS = {
  flap: ['Space', 'ArrowUp'],
  pause: ['KeyP'],
  restart: ['KeyR']
};

function isKeyEventLike(value) {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof value.code === 'string'
  );
}

export function createInput(bindings = DEFAULT_BINDINGS) {
  if (typeof bindings !== 'object' || bindings === null || Array.isArray(bindings)) {
    throw new TypeError('bindings must be an object');
  }
  const codeToAction = new Map();
  const seen = new Set();
  for (const action of Object.keys(bindings)) {
    if (!KNOWN_ACTIONS.includes(action)) {
      throw new TypeError(`unknown binding action: ${action}`);
    }
    const codes = bindings[action];
    if (!Array.isArray(codes)) {
      throw new TypeError(`binding ${action} must be an array of codes`);
    }
    for (const code of codes) {
      if (typeof code !== 'string' || code.length === 0) {
        throw new TypeError(`binding ${action} codes must be nonempty strings`);
      }
      if (seen.has(code)) {
        throw new TypeError(`duplicate binding code: ${code}`);
      }
      seen.add(code);
      codeToAction.set(code, action);
    }
  }

  let disposed = false;
  let held = new Set();
  let queue = [];

  function release(event) {
    if (disposed) {
      return false;
    }
    if (!isKeyEventLike(event)) {
      throw new TypeError('release expects an event with a code string');
    }
    held.delete(event.code);
    return codeToAction.has(event.code);
  }

  function handle(event) {
    if (disposed) {
      return false;
    }
    if (!isKeyEventLike(event)) {
      throw new TypeError('handle expects an event with a code string');
    }
    const type = event.type === undefined ? 'keydown' : event.type;
    if (type === 'keyup') {
      return release(event);
    }
    if (event.repeat === true) {
      return codeToAction.has(event.code);
    }
    if (held.has(event.code)) {
      return codeToAction.has(event.code);
    }
    const action = codeToAction.get(event.code);
    if (action === undefined) {
      return false;
    }
    held.add(event.code);
    queue.push(action);
    return true;
  }

  function consume() {
    const actions = queue;
    queue = [];
    return actions;
  }

  function reset() {
    held = new Set();
    queue = [];
  }

  function dispose() {
    disposed = true;
    held = new Set();
    queue = [];
  }

  return {handle, release, consume, reset, dispose};
}

function requireFinite(name, value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${name} must be a finite number`);
  }
}

function requirePositive(name, value) {
  requireFinite(name, value);
  if (value <= 0) {
    throw new RangeError(`${name} must be positive`);
  }
}

export function normalizePointer(event, rect, width, height) {
  if (typeof event !== 'object' || event === null) {
    throw new TypeError('event must be an object');
  }
  requireFinite('clientX', event.clientX);
  requireFinite('clientY', event.clientY);
  if (typeof rect !== 'object' || rect === null) {
    throw new TypeError('rect must be an object');
  }
  requireFinite('rect.left', rect.left);
  requireFinite('rect.top', rect.top);
  requirePositive('rect.width', rect.width);
  requirePositive('rect.height', rect.height);
  requirePositive('width', width);
  requirePositive('height', height);
  const x = ((event.clientX - rect.left) / rect.width) * width;
  const y = ((event.clientY - rect.top) / rect.height) * height;
  return {
    x: Math.min(width, Math.max(0, x)),
    y: Math.min(height, Math.max(0, y))
  };
}

export function createTicker(step, options = {}) {
  if (typeof step !== 'function') {
    throw new TypeError('step must be a function');
  }
  if (typeof options !== 'object' || options === null || Array.isArray(options)) {
    throw new TypeError('options must be an object');
  }
  const hz = options.hz === undefined ? 60 : options.hz;
  const maxSteps = options.maxSteps === undefined ? 5 : options.maxSteps;
  if (
    typeof hz !== 'number' ||
    !Number.isFinite(hz) ||
    hz <= 0 ||
    hz > 240
  ) {
    throw new RangeError('hz must be a positive finite number at most 240');
  }
  if (!Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > 100) {
    throw new RangeError('maxSteps must be an integer from 1 to 100');
  }

  const dtMs = 1000 / hz;
  let base = null;
  let paused = false;

  function update(timestamp) {
    if (typeof timestamp !== 'number' || !Number.isFinite(timestamp)) {
      throw new TypeError('timestamp must be a finite number');
    }
    if (paused) {
      return;
    }
    if (base === null) {
      base = timestamp;
      return;
    }
    if (timestamp < base) {
      throw new RangeError('timestamp must not decrease');
    }
    const elapsed = timestamp - base;
    let count = Math.floor(elapsed / dtMs + 1e-9);
    if (count > maxSteps) {
      count = maxSteps;
      base = timestamp;
      for (let i = 0; i < count; i += 1) step(dtMs / 1000);
      return;
    }
    if (count > 0) {
      base += count * dtMs;
      for (let i = 0; i < count; i += 1) {
        step(dtMs / 1000);
      }
    }
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

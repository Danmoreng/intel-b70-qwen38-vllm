// Public API contract: specs/input.md

const KNOWN_ACTIONS = ['flap', 'pause', 'restart'];
const DEFAULT_BINDINGS = {flap: ['Space', 'ArrowUp'], pause: ['KeyP'], restart: ['KeyR']};

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

export function createInput(bindings = DEFAULT_BINDINGS) {
  if (bindings === null || typeof bindings !== 'object' || Array.isArray(bindings)) {
    throw new TypeError('bindings must be an object mapping actions to code arrays');
  }
  const map = new Map(); // code -> action
  for (const action of Object.keys(bindings)) {
    if (!KNOWN_ACTIONS.includes(action)) {
      throw new Error(`unknown binding action: ${action}`);
    }
    const codes = bindings[action];
    if (!Array.isArray(codes)) {
      throw new TypeError(`binding "${action}" must be an array of key codes`);
    }
    for (const code of codes) {
      if (typeof code !== 'string' || code.length === 0) {
        throw new TypeError(`binding "${action}" codes must be nonempty strings`);
      }
      if (map.has(code)) {
        throw new Error(`duplicate binding code: ${code}`);
      }
      map.set(code, action);
    }
  }

  let held = new Set();
  let queue = [];
  let disposed = false;

  function checkEvent(event) {
    if (event === null || typeof event !== 'object') {
      throw new TypeError('expected a key-event-like object');
    }
  }

  function handle(event) {
    if (disposed) {
      return false;
    }
    checkEvent(event);
    const type = event.type === undefined ? 'keydown' : event.type;
    if (type === 'keyup') {
      return release(event);
    }
    if (type !== 'keydown') {
      return false;
    }
    const code = event.code;
    if (typeof code !== 'string') {
      return false;
    }
    const action = map.get(code);
    if (action === undefined) {
      return false;
    }
    if (event.repeat === true || held.has(code)) {
      return true;
    }
    held.add(code);
    queue.push(action);
    return true;
  }

  function release(event) {
    if (disposed) {
      return false;
    }
    checkEvent(event);
    const code = event.code;
    if (typeof code !== 'string') {
      return false;
    }
    held.delete(code);
    return map.has(code);
  }

  function consume() {
    if (disposed) {
      return [];
    }
    const queued = queue;
    queue = [];
    return queued;
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

export function normalizePointer(event, rect, width, height) {
  if (event === null || typeof event !== 'object') {
    throw new TypeError('normalizePointer expects a pointer-event-like object');
  }
  const {clientX, clientY} = event;
  if (!isFiniteNumber(clientX) || !isFiniteNumber(clientY)) {
    throw new RangeError('clientX and clientY must be finite numbers');
  }
  if (rect === null || typeof rect !== 'object') {
    throw new TypeError('normalizePointer expects a rect object');
  }
  const {left, top, width: rectWidth, height: rectHeight} = rect;
  if (
    !isFiniteNumber(left) ||
    !isFiniteNumber(top) ||
    !isFiniteNumber(rectWidth) ||
    !isFiniteNumber(rectHeight) ||
    rectWidth <= 0 ||
    rectHeight <= 0
  ) {
    throw new RangeError('rect must have finite offsets and positive dimensions');
  }
  if (!isFiniteNumber(width) || width <= 0 || !isFiniteNumber(height) || height <= 0) {
    throw new RangeError('logical width and height must be positive finite numbers');
  }
  const x = Math.min(width, Math.max(0, ((clientX - left) / rectWidth) * width));
  const y = Math.min(height, Math.max(0, ((clientY - top) / rectHeight) * height));
  return {x, y};
}

export function createTicker(step, options = {}) {
  if (typeof step !== 'function') {
    throw new TypeError('createTicker expects a step function');
  }
  const opts = options === null ? {} : options;
  if (opts !== null && typeof opts !== 'object') {
    throw new TypeError('ticker options must be an object');
  }
  const hz = opts.hz === undefined ? 60 : opts.hz;
  const maxSteps = opts.maxSteps === undefined ? 5 : opts.maxSteps;
  if (!isFiniteNumber(hz) || hz <= 0 || hz > 240) {
    throw new RangeError('hz must be a positive finite number <= 240');
  }
  if (typeof maxSteps !== 'number' || !Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > 100) {
    throw new RangeError('maxSteps must be an integer between 1 and 100');
  }
  const dt = 1000 / hz;

  let base = null;
  let remainder = 0;
  let paused = false;

  function update(timestampMs) {
    if (!isFiniteNumber(timestampMs)) {
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
      throw new RangeError('timestamp must not decrease');
    }
    remainder += timestampMs - base;
    base = timestampMs;
    let steps = 0;
    while (remainder >= dt - 1e-9 && steps < maxSteps) {
      step(dt / 1000);
      remainder -= dt;
      steps++;
    }
    if (steps === maxSteps && remainder >= dt) {
      // Drop excess backlog but retain the fractional remainder.
      remainder %= dt;
    }
    if (Math.abs(remainder) < 1e-9) {
      remainder = 0;
    }
  }

  function pause() {
    paused = true;
    base = null;
    remainder = 0;
  }

  function resume() {
    paused = false;
    base = null;
    remainder = 0;
  }

  function reset() {
    base = null;
    remainder = 0;
  }

  return {update, pause, resume, reset};
}

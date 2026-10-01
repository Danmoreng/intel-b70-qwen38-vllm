// Implementation of specs/input.md — edge-triggered key bindings, pointer
// mapping and a bounded fixed-step ticker. Pure module: no global DOM
// listeners, no requestAnimationFrame, no wall clock.

const KNOWN_ACTIONS = Object.freeze(['flap', 'pause', 'restart']);
const DEFAULT_BINDINGS = Object.freeze({
  flap: Object.freeze(['Space', 'ArrowUp']),
  pause: Object.freeze(['KeyP']),
  restart: Object.freeze(['KeyR']),
});

const isFiniteNum = (v) => typeof v === 'number' && Number.isFinite(v);

function bindingMap(bindings) {
  if (typeof bindings !== 'object' || bindings === null || Array.isArray(bindings)) {
    throw new TypeError('bindings must be an object of code arrays');
  }
  for (const key of Object.keys(bindings)) {
    if (!KNOWN_ACTIONS.includes(key)) {
      throw new TypeError(`unknown binding action: ${key}`);
    }
  }
  const map = new Map();
  for (const action of KNOWN_ACTIONS) {
    if (!Object.prototype.hasOwnProperty.call(bindings, action)) continue;
    const codes = bindings[action];
    if (!Array.isArray(codes)) {
      throw new TypeError(`binding ${action} must be an array of codes`);
    }
    for (const code of codes) {
      if (typeof code !== 'string' || code.length === 0) {
        throw new TypeError('binding codes must be nonempty strings');
      }
      if (map.has(code)) {
        throw new TypeError(`duplicate binding code: ${code}`);
      }
      map.set(code, action);
    }
  }
  return map;
}

export function createInput(bindings = DEFAULT_BINDINGS) {
  const actionFor = bindingMap(bindings);
  let held = new Set();
  let queue = [];
  let disposed = false;

  const recognized = (event) =>
    event && typeof event.code === 'string' && actionFor.has(event.code);

  function handle(event) {
    if (disposed) return false;
    const type = !event || event.type === undefined ? 'keydown' : event.type;
    if (type === 'keyup') return release(event);
    if (type !== 'keydown') return false;
    if (!recognized(event)) return false;
    if (event.repeat === true || held.has(event.code)) return true; // recognized, no new edge
    held.add(event.code);
    queue.push(actionFor.get(event.code));
    return true;
  }

  function release(event) {
    if (disposed) return false;
    if (!recognized(event)) return false;
    held.delete(event.code);
    return true; // recognized codes return true even when not held
  }

  function consume() {
    if (disposed) return [];
    const out = queue;
    queue = [];
    return out;
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
  if (!event || !isFiniteNum(event.clientX) || !isFiniteNum(event.clientY)) {
    throw new TypeError('event must have finite clientX/clientY');
  }
  if (!rect || !isFiniteNum(rect.left) || !isFiniteNum(rect.top) ||
      !isFiniteNum(rect.width) || rect.width <= 0 ||
      !isFiniteNum(rect.height) || rect.height <= 0) {
    throw new TypeError('rect must have finite left/top and positive width/height');
  }
  if (!isFiniteNum(width) || width <= 0 || !isFiniteNum(height) || height <= 0) {
    throw new RangeError('logical width/height must be positive');
  }
  const x = Math.min(Math.max(((event.clientX - rect.left) / rect.width) * width, 0), width);
  const y = Math.min(Math.max(((event.clientY - rect.top) / rect.height) * height, 0), height);
  return {x, y};
}

export function createTicker(step, options = {hz: 60, maxSteps: 5}) {
  if (typeof step !== 'function') {
    throw new TypeError('step must be a function');
  }
  if (typeof options !== 'object' || options === null) {
    throw new TypeError('ticker options must be an object');
  }
  const hz = options.hz === undefined ? 60 : options.hz;
  const maxSteps = options.maxSteps === undefined ? 5 : options.maxSteps;
  if (!isFiniteNum(hz) || hz <= 0 || hz > 240) {
    throw new RangeError('hz must be a positive finite number <= 240');
  }
  if (!Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > 100) {
    throw new RangeError('maxSteps must be an integer 1..100');
  }
  const dtMs = 1000 / hz;
  const dt = dtMs / 1000;
  let paused = false;
  let base = null;

  function update(timestampMs) {
    if (typeof timestampMs !== 'number' || !Number.isFinite(timestampMs)) {
      throw new RangeError('timestamp must be a finite number');
    }
    if (paused) return 0; // paused: no callbacks, base already cleared
    if (base === null) {
      base = timestampMs; // first timestamp initializes
      return 0;
    }
    if (timestampMs < base) {
      throw new RangeError('timestamps must not decrease'); // atomic: nothing changed
    }
    const elapsed = timestampMs - base;
    const available = Math.floor(elapsed / dtMs);
    const run = Math.min(available, maxSteps);
    for (let i = 0; i < run; i += 1) step(dt);
    if (available > maxSteps) {
      base = timestampMs; // drop the excess backlog
    } else {
      base = timestampMs - (elapsed % dtMs); // retain fractional remainder
    }
    return run;
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

// Input and ticker API per specs/input.md.
// No global DOM listeners and no requestAnimationFrame internally.

const KNOWN_ACTIONS = ['flap', 'pause', 'restart'];

function isFiniteNumber(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function asPlainObject(v) {
  if (v === null || typeof v !== 'object' || Array.isArray(v)) {
    throw new TypeError('argument must be a non-null, non-array object');
  }
  return v;
}

export function createInput(bindings = {flap: ['Space', 'ArrowUp'], pause: ['KeyP'], restart: ['KeyR']}) {
  const map = asPlainObject(bindings);
  const codeToAction = new Map();
  for (const action of Object.keys(map)) {
    if (!KNOWN_ACTIONS.includes(action)) {
      throw new TypeError(`unknown action binding: ${action}`);
    }
    const codes = map[action];
    if (!Array.isArray(codes)) {
      throw new TypeError(`binding for ${action} must be an array of codes`);
    }
    for (const code of codes) {
      if (typeof code !== 'string' || code.length === 0) {
        throw new TypeError('binding codes must be nonempty strings');
      }
      if (codeToAction.has(code)) {
        throw new TypeError(`duplicate binding code: ${code}`);
      }
      codeToAction.set(code, action);
    }
  }

  const held = new Set();
  const queue = [];
  let disposed = false;

  function isEvent(event) {
    return event !== null && typeof event === 'object' && typeof event.code === 'string';
  }

  function release(event) {
    if (disposed || !isEvent(event)) {
      return false;
    }
    const {code} = event;
    if (!codeToAction.has(code)) {
      return false;
    }
    held.delete(code);
    return true;
  }

  function handle(event) {
    if (disposed || event === null || typeof event !== 'object') {
      return false;
    }
    const type = event.type === undefined ? 'keydown' : event.type;
    if (type === 'keyup') {
      return release(event);
    }
    if (type !== 'keydown' || !isEvent(event)) {
      return false;
    }
    const {code} = event;
    const action = codeToAction.get(code);
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

  return {
    handle,
    release,
    consume() {
      if (disposed) {
        return [];
      }
      return queue.splice(0, queue.length);
    },
    reset() {
      held.clear();
      queue.length = 0;
    },
    dispose() {
      disposed = true;
      held.clear();
      queue.length = 0;
    },
  };
}

export function normalizePointer(event, rect, width, height) {
  if (event === null || typeof event !== 'object' ||
      !isFiniteNumber(event.clientX) || !isFiniteNumber(event.clientY) ||
      rect === null || typeof rect !== 'object' ||
      !isFiniteNumber(rect.left) || !isFiniteNumber(rect.top) ||
      !isFiniteNumber(rect.width) || !isFiniteNumber(rect.height) ||
      rect.width <= 0 || rect.height <= 0 ||
      !isFiniteNumber(width) || !isFiniteNumber(height) || width <= 0 || height <= 0) {
    throw new TypeError('normalizePointer requires finite clientX/clientY, a finite rect with positive dimensions, and positive logical dimensions');
  }
  const x = ((event.clientX - rect.left) * width) / rect.width;
  const y = ((event.clientY - rect.top) * height) / rect.height;
  return {
    x: Math.min(Math.max(x, 0), width),
    y: Math.min(Math.max(y, 0), height),
  };
}

export function createTicker(step, options = {hz: 60, maxSteps: 5}) {
  const opts = asPlainObject(options);
  const hz = opts.hz === undefined ? 60 : opts.hz;
  const maxSteps = opts.maxSteps === undefined ? 5 : opts.maxSteps;
  if (!isFiniteNumber(hz) || hz <= 0 || hz > 240) {
    throw new RangeError('hz must be a positive finite number at most 240');
  }
  if (!Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > 100) {
    throw new RangeError('maxSteps must be an integer between 1 and 100');
  }
  const dtMs = 1000 / hz;
  const dtSec = dtMs / 1000;
  let last = null;
  let acc = 0;
  let paused = false;

  return {
    update(timestampMs) {
      if (!isFiniteNumber(timestampMs)) {
        throw new TypeError('timestamp must be a finite number');
      }
      if (paused) {
        return 0;
      }
      if (last === null) {
        last = timestampMs;
        return 0;
      }
      if (timestampMs < last) {
        throw new RangeError('timestamp must not decrease');
      }
      acc += timestampMs - last;
      last = timestampMs;
      const steps = Math.floor(acc / dtMs);
      const n = Math.min(steps, maxSteps);
      for (let i = 0; i < n; i++) {
        step(dtSec);
      }
      acc = steps > maxSteps ? 0 : acc - n * dtMs;
      return n;
    },
    pause() {
      paused = true;
      last = null;
      acc = 0;
    },
    resume() {
      paused = false;
      last = null;
      acc = 0;
    },
    reset() {
      paused = false;
      last = null;
      acc = 0;
    },
  };
}

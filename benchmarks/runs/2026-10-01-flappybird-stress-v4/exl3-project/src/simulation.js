// Public API contract: specs/simulation.md
import {validateConfig} from './config.js';
import {createRng} from './random.js';

const PHASES = new Set(['ready', 'running', 'paused', 'gameover']);

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function isUint32(value) {
  return (
    typeof value === 'number' &&
    Number.isInteger(value) &&
    value >= 0 &&
    value <= 0xffffffff
  );
}

function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function createGame(config = {}) {
  const validated = validateConfig(config);
  return {
    version: 1,
    config: validated,
    phase: 'ready',
    frame: 0,
    score: 0,
    rngState: createRng(validated.seed).snapshot(),
    bird: {
      x: validated.birdX,
      y: (validated.height - validated.groundHeight) / 2,
      vy: 0
    },
    pipes: []
  };
}

export function cloneState(state) {
  return {
    version: state.version,
    config: {...state.config},
    phase: state.phase,
    frame: state.frame,
    score: state.score,
    rngState: state.rngState,
    bird: {...state.bird},
    pipes: state.pipes.map((pipe) => ({...pipe}))
  };
}

export function flap(state) {
  const next = cloneState(state);
  if (next.phase === 'ready') {
    next.phase = 'running';
  }
  if (next.phase === 'running') {
    next.bird.vy = next.config.flapVelocity;
  }
  return next;
}

export function pause(state) {
  const next = cloneState(state);
  if (next.phase === 'running') {
    next.phase = 'paused';
  }
  return next;
}

export function resume(state) {
  const next = cloneState(state);
  if (next.phase === 'paused') {
    next.phase = 'running';
  }
  return next;
}

export function restart(state) {
  return createGame(state.config);
}

export function tick(state, input = {}) {
  if (state.phase !== 'running') {
    return cloneState(state);
  }

  const config = state.config;
  const frame = state.frame + 1;

  let vy = input.flap === true ? config.flapVelocity : state.bird.vy;
  vy += config.gravity;
  const bird = {x: config.birdX, y: state.bird.y + vy, vy};

  const pipes = state.pipes.map((pipe) => ({...pipe, x: pipe.x - config.pipeSpeed}));

  let rngState = state.rngState;
  if (frame % config.spawnEvery === 0) {
    const rng = createRng(rngState);
    const roll = rng.next();
    rngState = rng.snapshot();
    const span = config.height - config.groundHeight - config.gapHeight - 40 + 1;
    pipes.push({
      id: frame,
      x: config.width,
      gapY: 20 + Math.floor(roll * span),
      passed: false
    });
  }

  const r = config.birdRadius;
  let collided =
    bird.y - r <= 0 || bird.y + r >= config.height - config.groundHeight;
  if (!collided) {
    for (const pipe of pipes) {
      const overlap =
        bird.x + r >= pipe.x && bird.x - r <= pipe.x + config.pipeWidth;
      if (
        overlap &&
        (bird.y - r <= pipe.gapY || bird.y + r >= pipe.gapY + config.gapHeight)
      ) {
        collided = true;
        break;
      }
    }
  }

  let score = state.score;
  let phase;
  if (collided) {
    phase = 'gameover';
  } else {
    phase = 'running';
    for (const pipe of pipes) {
      if (!pipe.passed && pipe.x + config.pipeWidth < bird.x - r) {
        pipe.passed = true;
        score += 1;
      }
    }
  }

  return {
    version: state.version,
    config: {...config},
    phase,
    frame,
    score,
    rngState,
    bird,
    pipes: pipes.filter((pipe) => pipe.x + config.pipeWidth >= 0)
  };
}

export function restoreState(snapshot) {
  if (!isPlainObject(snapshot)) {
    throw new TypeError('snapshot must be an object');
  }
  if (snapshot.version !== 1) {
    throw new TypeError('snapshot version must be 1');
  }
  if (!PHASES.has(snapshot.phase)) {
    throw new TypeError('snapshot phase is not a legal phase');
  }
  if (!Number.isInteger(snapshot.frame) || snapshot.frame < 0) {
    throw new TypeError('snapshot frame must be a nonnegative integer');
  }
  if (!Number.isInteger(snapshot.score) || snapshot.score < 0) {
    throw new TypeError('snapshot score must be a nonnegative integer');
  }
  if (!isUint32(snapshot.rngState) || snapshot.rngState === 0) {
    throw new TypeError('snapshot rngState must be a nonzero uint32');
  }
  validateConfig(snapshot.config);
  if (!isPlainObject(snapshot.bird)) {
    throw new TypeError('snapshot bird must be an object');
  }
  for (const key of ['x', 'y', 'vy']) {
    if (!isFiniteNumber(snapshot.bird[key])) {
      throw new TypeError(`snapshot bird ${key} must be finite`);
    }
  }
  if (!Array.isArray(snapshot.pipes)) {
    throw new TypeError('snapshot pipes must be an array');
  }
  const seenIds = new Set();
  for (const pipe of snapshot.pipes) {
    if (!isPlainObject(pipe)) {
      throw new TypeError('each pipe must be an object');
    }
    if (!Number.isInteger(pipe.id) || pipe.id < 0) {
      throw new TypeError('pipe id must be a nonnegative integer');
    }
    if (seenIds.has(pipe.id)) {
      throw new TypeError('pipe ids must be unique');
    }
    seenIds.add(pipe.id);
    if (!isFiniteNumber(pipe.x)) {
      throw new TypeError('pipe x must be finite');
    }
    if (!isFiniteNumber(pipe.gapY)) {
      throw new TypeError('pipe gapY must be finite');
    }
    if (typeof pipe.passed !== 'boolean') {
      throw new TypeError('pipe passed must be a boolean');
    }
  }
  return cloneState(snapshot);
}

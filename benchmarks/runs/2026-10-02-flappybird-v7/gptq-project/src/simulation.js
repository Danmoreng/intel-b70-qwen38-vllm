// Deterministic fixed-step flight simulation, per specs/simulation.md.
// Pure module: every transition is pure and returns detached objects.

import {createRng} from './random.js';
import {validateConfig} from './config.js';

const PHASES = new Set(['ready', 'running', 'paused', 'gameover']);
const MAX_UINT32 = 0xffffffff;

export function createGame(config = {}) {
  const cfg = validateConfig(config);
  return {
    version: 1,
    config: cfg,
    phase: 'ready',
    frame: 0,
    score: 0,
    rngState: createRng(cfg.seed).snapshot(),
    bird: {x: cfg.birdX, y: (cfg.height - cfg.groundHeight) / 2, vy: 0},
    pipes: [],
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
    pipes: state.pipes.map((pipe) => ({...pipe})),
  };
}

export function restart(state) {
  return createGame(state.config);
}

function detach(state) {
  return cloneState(state);
}

export function flap(state) {
  const next = detach(state);
  if (next.phase === 'ready' || next.phase === 'running') {
    next.phase = 'running';
    next.bird.vy = next.config.flapVelocity;
  }
  return next;
}

export function pause(state) {
  const next = detach(state);
  if (state.phase === 'running') {
    next.phase = 'paused';
  }
  return next;
}

export function resume(state) {
  const next = detach(state);
  if (state.phase === 'paused') {
    next.phase = 'running';
  }
  return next;
}

function hitsPipe(bird, r, pipe, cfg) {
  const top = pipe.gapY;
  const bottom = pipe.gapY + cfg.gapHeight;
  return bird.x + r >= pipe.x &&
    bird.x - r <= pipe.x + cfg.pipeWidth &&
    (bird.y - r <= top || bird.y + r >= bottom);
}

export function tick(state, input = {}) {
  const next = detach(state);
  if (next.phase !== 'running') {
    return next;
  }

  const cfg = next.config;
  next.frame += 1;

  const bird = next.bird;
  if (input.flap === true) {
    bird.vy = cfg.flapVelocity;
  }
  bird.vy += cfg.gravity;
  bird.y += bird.vy;

  const pipes = next.pipes;
  for (const pipe of pipes) {
    pipe.x -= cfg.pipeSpeed;
  }
  if (next.frame % cfg.spawnEvery === 0) {
    const rng = createRng(0);
    rng.restore(next.rngState);
    const span = cfg.height - cfg.groundHeight - cfg.gapHeight - 40 + 1;
    const gapY = 20 + Math.floor(rng.next() * span);
    pipes.push({id: next.frame, x: cfg.width, gapY, passed: false});
    next.rngState = rng.snapshot();
  }

  const r = cfg.birdRadius;
  let hit = bird.y - r <= 0 || bird.y + r >= cfg.height - cfg.groundHeight;
  if (!hit) {
    for (const pipe of pipes) {
      if (hitsPipe(bird, r, pipe, cfg)) {
        hit = true;
        break;
      }
    }
  }

  if (hit) {
    next.phase = 'gameover';
  } else {
    for (const pipe of pipes) {
      if (!pipe.passed && pipe.x + cfg.pipeWidth < bird.x - r) {
        pipe.passed = true;
        next.score += 1;
      }
    }
  }

  next.pipes = pipes.filter((pipe) => pipe.x + cfg.pipeWidth >= 0);
  return next;
}

function isNonNegativeInteger(v) {
  return typeof v === 'number' && Number.isInteger(v) && v >= 0;
}

function isFiniteNumber(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

export function restoreState(snapshot) {
  if (snapshot === null || typeof snapshot !== 'object' || Array.isArray(snapshot)) {
    throw new TypeError('restoreState accepts only a state snapshot object');
  }
  if (snapshot.version !== 1) {
    throw new TypeError('snapshot version must be 1');
  }
  const config = validateConfig(snapshot.config);
  const {phase, frame, score, rngState, bird, pipes} = snapshot;
  if (typeof phase !== 'string' || !PHASES.has(phase)) {
    throw new TypeError(`illegal phase "${phase}"`);
  }
  if (!isNonNegativeInteger(frame) || !isNonNegativeInteger(score)) {
    throw new TypeError('frame and score must be nonnegative integers');
  }
  if (typeof rngState !== 'number' || !Number.isInteger(rngState) ||
    rngState < 1 || rngState > MAX_UINT32) {
    throw new TypeError('rngState must be a nonzero uint32');
  }
  if (bird === null || typeof bird !== 'object' || Array.isArray(bird)) {
    throw new TypeError('bird must be an object');
  }
  for (const key of ['x', 'y', 'vy']) {
    if (!isFiniteNumber(bird[key])) {
      throw new TypeError(`bird.${key} must be a finite number`);
    }
  }
  if (!Array.isArray(pipes)) {
    throw new TypeError('pipes must be an array');
  }
  const seen = new Set();
  for (const pipe of pipes) {
    if (pipe === null || typeof pipe !== 'object' || Array.isArray(pipe)) {
      throw new TypeError('each pipe must be an object');
    }
    if (!isNonNegativeInteger(pipe.id) || seen.has(pipe.id)) {
      throw new TypeError('pipe ids must be unique nonnegative integers');
    }
    seen.add(pipe.id);
    if (!isFiniteNumber(pipe.x) || !isFiniteNumber(pipe.gapY)) {
      throw new TypeError('pipe x and gapY must be finite numbers');
    }
    if (typeof pipe.passed !== 'boolean') {
      throw new TypeError('pipe passed must be a boolean');
    }
  }
  return {
    version: 1,
    config,
    phase,
    frame,
    score,
    rngState,
    bird: {...bird},
    pipes: pipes.map((pipe) => ({...pipe})),
  };
}

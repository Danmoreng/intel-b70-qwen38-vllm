// Public API contract: specs/simulation.md
import {createRng} from './random.js';
import {validateConfig} from './config.js';

const PHASES = ['ready', 'running', 'paused', 'gameover'];
const ZERO_SEED_STATE = 0x6d2b79f5;
const MAX_UINT32 = 0xFFFFFFFF;

function cloneValue(value) {
  if (Array.isArray(value)) {
    return value.map(cloneValue);
  }
  if (value !== null && typeof value === 'object') {
    const out = {};
    for (const key of Object.keys(value)) {
      out[key] = cloneValue(value[key]);
    }
    return out;
  }
  return value;
}

function isNonnegInt(value) {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}

function isUint32(value) {
  return isNonnegInt(value) && value <= MAX_UINT32;
}

function ownKeys(value) {
  return Object.keys(value);
}

function hasOnlyKeys(value, allowed) {
  return ownKeys(value).every((key) => allowed.includes(key));
}

export function cloneState(state) {
  return cloneValue(state);
}

export function createGame(config = {}) {
  const cfg = validateConfig(config);
  return {
    version: 1,
    config: cloneValue(cfg),
    phase: 'ready',
    frame: 0,
    score: 0,
    rngState: cfg.seed === 0 ? ZERO_SEED_STATE : cfg.seed,
    bird: {x: cfg.birdX, y: (cfg.height - cfg.groundHeight) / 2, vy: 0},
    pipes: []
  };
}

export function flap(state) {
  const next = cloneState(state);
  if (next.phase === 'ready' || next.phase === 'running') {
    next.phase = 'running';
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
  const next = cloneState(state);
  if (next.phase !== 'running') {
    return next;
  }
  const cfg = next.config;
  next.frame += 1;
  if (input && input.flap === true) {
    next.bird.vy = cfg.flapVelocity;
  }
  next.bird.vy += cfg.gravity;
  next.bird.y += next.bird.vy;

  for (const pipe of next.pipes) {
    pipe.x -= cfg.pipeSpeed;
  }

  let rngState = next.rngState;
  if (next.frame % cfg.spawnEvery === 0) {
    const rng = createRng(rngState);
    const span = cfg.height - cfg.groundHeight - cfg.gapHeight - 40 + 1;
    const gapY = 20 + Math.floor(rng.next() * span);
    next.pipes.push({id: next.frame, x: cfg.width, gapY, passed: false});
    rngState = rng.snapshot();
  }
  next.rngState = rngState;

  const bird = next.bird;
  const r = cfg.birdRadius;
  let collided = bird.y - r <= 0 || bird.y + r >= cfg.height - cfg.groundHeight;
  if (!collided) {
    for (const pipe of next.pipes) {
      const overlapsX = bird.x + r >= pipe.x && bird.x - r <= pipe.x + cfg.pipeWidth;
      if (overlapsX && (bird.y - r <= pipe.gapY || bird.y + r >= pipe.gapY + cfg.gapHeight)) {
        collided = true;
        break;
      }
    }
  }

  if (collided) {
    next.phase = 'gameover';
  } else {
    for (const pipe of next.pipes) {
      if (!pipe.passed && pipe.x + cfg.pipeWidth < bird.x - r) {
        pipe.passed = true;
        next.score += 1;
      }
    }
  }

  next.pipes = next.pipes.filter((pipe) => pipe.x + cfg.pipeWidth >= 0);
  return next;
}

export function restoreState(snapshot) {
  if (snapshot === null || typeof snapshot !== 'object' || Array.isArray(snapshot)) {
    throw new TypeError('restoreState expects a state snapshot object');
  }
  if (!hasOnlyKeys(snapshot, ['version', 'config', 'phase', 'frame', 'score', 'rngState', 'bird', 'pipes'])) {
    throw new Error('restoreState snapshot has unknown keys');
  }
  if (snapshot.version !== 1) {
    throw new Error('restoreState snapshot must have version 1');
  }
  // validateConfig throws on null/unknown/bad values and returns a detached copy.
  validateConfig(snapshot.config);
  if (!PHASES.includes(snapshot.phase)) {
    throw new Error('restoreState snapshot phase is invalid');
  }
  if (!isNonnegInt(snapshot.frame)) {
    throw new Error('restoreState snapshot frame must be a nonnegative integer');
  }
  if (!isNonnegInt(snapshot.score)) {
    throw new Error('restoreState snapshot score must be a nonnegative integer');
  }
  if (!isUint32(snapshot.rngState) || snapshot.rngState === 0) {
    throw new Error('restoreState snapshot rngState must be a nonzero uint32');
  }
  if (
    snapshot.bird === null ||
    typeof snapshot.bird !== 'object' ||
    !hasOnlyKeys(snapshot.bird, ['x', 'y', 'vy'])
  ) {
    throw new Error('restoreState snapshot bird is invalid');
  }
  for (const key of ['x', 'y', 'vy']) {
    if (typeof snapshot.bird[key] !== 'number' || !Number.isFinite(snapshot.bird[key])) {
      throw new Error(`restoreState snapshot bird ${key} must be a finite number`);
    }
  }
  if (!Array.isArray(snapshot.pipes)) {
    throw new Error('restoreState snapshot pipes must be an array');
  }
  const seenIds = new Set();
  for (const pipe of snapshot.pipes) {
    if (
      pipe === null ||
      typeof pipe !== 'object' ||
      !hasOnlyKeys(pipe, ['id', 'x', 'gapY', 'passed'])
    ) {
      throw new Error('restoreState snapshot pipe is invalid');
    }
    if (!isNonnegInt(pipe.id) || seenIds.has(pipe.id)) {
      throw new Error('restoreState snapshot pipe id must be a unique nonnegative integer');
    }
    seenIds.add(pipe.id);
    if (typeof pipe.x !== 'number' || !Number.isFinite(pipe.x)) {
      throw new Error('restoreState snapshot pipe x must be a finite number');
    }
    if (typeof pipe.gapY !== 'number' || !Number.isFinite(pipe.gapY)) {
      throw new Error('restoreState snapshot pipe gapY must be a finite number');
    }
    if (typeof pipe.passed !== 'boolean') {
      throw new Error('restoreState snapshot pipe passed must be a boolean');
    }
  }
  return cloneState(snapshot);
}

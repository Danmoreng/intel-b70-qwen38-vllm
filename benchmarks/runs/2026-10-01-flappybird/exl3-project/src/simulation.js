// Deterministic fixed-step simulation per specs/simulation.md.
// Pure transitions, detached returns, no wall clock or DOM.

import {createRng} from './random.js';
import {validateConfig} from './config.js';

const PHASES = new Set(['ready', 'running', 'paused', 'gameover']);
const MAX_UINT32 = 0xffffffff;

function deepClone(value) {
  if (Array.isArray(value)) {
    return value.map(deepClone);
  }
  if (value !== null && typeof value === 'object') {
    const out = {};
    for (const key of Object.keys(value)) {
      out[key] = deepClone(value[key]);
    }
    return out;
  }
  return value;
}

export function cloneState(state) {
  return deepClone(state);
}

export function createGame(config = {}) {
  const cfg = validateConfig(config);
  const rng = createRng(cfg.seed);
  return {
    version: 1,
    config: cfg,
    phase: 'ready',
    frame: 0,
    score: 0,
    rngState: rng.snapshot(),
    bird: {x: cfg.birdX, y: (cfg.height - cfg.groundHeight) / 2, vy: 0},
    pipes: [],
  };
}

export function flap(state) {
  const s = cloneState(state);
  if (s.phase === 'ready') {
    s.phase = 'running';
    s.bird.vy = s.config.flapVelocity;
  } else if (s.phase === 'running') {
    s.bird.vy = s.config.flapVelocity;
  }
  return s;
}

export function tick(state, input = {}) {
  const s = cloneState(state);
  if (s.phase !== 'running') {
    return s;
  }
  const cfg = s.config;
  s.frame += 1;
  if (input.flap === true) {
    s.bird.vy = cfg.flapVelocity;
  }
  s.bird.vy += cfg.gravity;
  s.bird.y += s.bird.vy;
  for (const pipe of s.pipes) {
    pipe.x -= cfg.pipeSpeed;
  }
  if (s.frame % cfg.spawnEvery === 0) {
    const rng = createRng(0);
    rng.restore(s.rngState);
    const range = cfg.height - cfg.groundHeight - cfg.gapHeight - 40 + 1;
    const gapY = 20 + Math.floor(rng.next() * range);
    s.rngState = rng.snapshot();
    s.pipes.push({id: s.frame, x: cfg.width, gapY, passed: false});
  }
  const r = cfg.birdRadius;
  let hit = s.bird.y - r <= 0 || s.bird.y + r >= cfg.height - cfg.groundHeight;
  if (!hit) {
    for (const pipe of s.pipes) {
      const xOverlap = s.bird.x + r >= pipe.x && s.bird.x - r <= pipe.x + cfg.pipeWidth;
      const yOverlap = s.bird.y - r <= pipe.gapY || s.bird.y + r >= pipe.gapY + cfg.gapHeight;
      if (xOverlap && yOverlap) {
        hit = true;
        break;
      }
    }
  }
  if (hit) {
    s.phase = 'gameover';
  } else {
    for (const pipe of s.pipes) {
      if (!pipe.passed && pipe.x + cfg.pipeWidth < s.bird.x - r) {
        pipe.passed = true;
        s.score += 1;
      }
    }
  }
  s.pipes = s.pipes.filter(pipe => pipe.x + cfg.pipeWidth >= 0);
  return s;
}

export function pause(state) {
  const s = cloneState(state);
  if (s.phase === 'running') {
    s.phase = 'paused';
  }
  return s;
}

export function resume(state) {
  const s = cloneState(state);
  if (s.phase === 'paused') {
    s.phase = 'running';
  }
  return s;
}

export function restart(state) {
  return createGame(state.config);
}

function isNonNegativeInt(v) {
  return typeof v === 'number' && Number.isInteger(v) && v >= 0;
}

function isFiniteNumber(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function isPlainObject(v) {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

export function restoreState(snapshot) {
  if (!isPlainObject(snapshot)) {
    throw new TypeError('snapshot must be a non-null, non-array object');
  }
  if (snapshot.version !== 1) {
    throw new TypeError('snapshot version must be 1');
  }
  if (!isPlainObject(snapshot.config)) {
    throw new TypeError('snapshot config must be an object');
  }
  const config = validateConfig(snapshot.config);
  if (!PHASES.has(snapshot.phase)) {
    throw new TypeError('snapshot phase must be ready, running, paused or gameover');
  }
  if (!isNonNegativeInt(snapshot.frame) || !isNonNegativeInt(snapshot.score)) {
    throw new TypeError('snapshot frame and score must be nonnegative integers');
  }
  const rngState = snapshot.rngState;
  if (typeof rngState !== 'number' || !Number.isInteger(rngState) || rngState < 1 || rngState > MAX_UINT32) {
    throw new TypeError('snapshot rngState must be a nonzero unsigned 32-bit integer');
  }
  const bird = snapshot.bird;
  if (!isPlainObject(bird)) {
    throw new TypeError('snapshot bird must be an object');
  }
  for (const key of ['x', 'y', 'vy']) {
    if (!isFiniteNumber(bird[key])) {
      throw new TypeError(`snapshot bird ${key} must be a finite number`);
    }
  }
  if (!Array.isArray(snapshot.pipes)) {
    throw new TypeError('snapshot pipes must be an array');
  }
  const seenIds = new Set();
  for (const pipe of snapshot.pipes) {
    if (!isPlainObject(pipe)) {
      throw new TypeError('snapshot pipe entries must be objects');
    }
    if (!isNonNegativeInt(pipe.id) || seenIds.has(pipe.id)) {
      throw new TypeError('snapshot pipe IDs must be unique nonnegative integers');
    }
    seenIds.add(pipe.id);
    if (!isFiniteNumber(pipe.x) || !isFiniteNumber(pipe.gapY)) {
      throw new TypeError('snapshot pipe x and gapY must be finite numbers');
    }
    if (typeof pipe.passed !== 'boolean') {
      throw new TypeError('snapshot pipe passed must be a boolean');
    }
  }
  return {
    version: 1,
    config,
    phase: snapshot.phase,
    frame: snapshot.frame,
    score: snapshot.score,
    rngState,
    bird: {x: bird.x, y: bird.y, vy: bird.vy},
    pipes: snapshot.pipes.map(pipe => ({id: pipe.id, x: pipe.x, gapY: pipe.gapY, passed: pipe.passed})),
  };
}

// Public API contract: specs/simulation.md
import {createRng} from './random.js';
import {validateConfig} from './config.js';

const PHASES = new Set(['ready', 'running', 'paused', 'gameover']);

export function cloneState(state) {
  return structuredClone(state);
}

// New game from a validated config; RNG is initialized from the seed.
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
    pipes: []
  };
}

// flap: ready -> running, running re-flaps; paused/gameover is a detached no-op.
export function flap(state) {
  if (state.phase !== 'ready' && state.phase !== 'running') {
    return cloneState(state);
  }
  const s = cloneState(state);
  s.phase = 'running';
  s.bird.vy = state.config.flapVelocity;
  return s;
}

export function pause(state) {
  const s = cloneState(state);
  if (state.phase === 'running') {
    s.phase = 'paused';
  }
  return s;
}

export function resume(state) {
  const s = cloneState(state);
  if (state.phase === 'paused') {
    s.phase = 'running';
  }
  return s;
}

// restart: same config, fresh deterministic state.
export function restart(state) {
  return createGame(state.config);
}

function isInteger(v) {
  return Number.isInteger(v);
}

// Validates the complete snapshot shape; rejects atomically, returns detached.
export function restoreState(snapshot) {
  if (snapshot === null || typeof snapshot !== 'object' || Array.isArray(snapshot)) {
    throw new TypeError('snapshot must be an object');
  }
  const keys = Object.keys(snapshot).sort();
  const expected = ['bird', 'config', 'frame', 'phase', 'pipes', 'rngState', 'score', 'version'];
  if (keys.length !== expected.length || expected.some((k, i) => keys[i] !== k)) {
    throw new TypeError('snapshot has an unexpected shape');
  }
  const {version, config, phase, frame, score, rngState, bird, pipes} = snapshot;
  if (version !== 1) {
    throw new RangeError('snapshot version must be 1');
  }
  const cfg = validateConfig(config);
  if (!PHASES.has(phase)) {
    throw new RangeError(`snapshot phase must be legal, got ${String(phase)}`);
  }
  if (!isInteger(frame) || frame < 0) {
    throw new RangeError('snapshot frame must be a nonnegative integer');
  }
  if (!isInteger(score) || score < 0) {
    throw new RangeError('snapshot score must be a nonnegative integer');
  }
  if (typeof rngState !== 'number' || !isInteger(rngState) || rngState < 1 || rngState > 0xffffffff) {
    throw new RangeError('snapshot rngState must be a nonzero uint32');
  }
  if (bird === null || typeof bird !== 'object' || Array.isArray(bird)) {
    throw new TypeError('snapshot bird must be an object');
  }
  for (const k of ['x', 'y', 'vy']) {
    if (typeof bird[k] !== 'number' || !isFinite(bird[k])) {
      throw new RangeError(`snapshot bird.${k} must be finite`);
    }
  }
  for (const [k, v] of Object.entries(bird)) {
    if (typeof v === 'number' && !isFinite(v)) {
      throw new RangeError(`snapshot bird.${k} must be finite`);
    }
  }
  if (!Array.isArray(pipes)) {
    throw new TypeError('snapshot pipes must be an array');
  }
  const seen = new Set();
  for (const p of pipes) {
    if (p === null || typeof p !== 'object' || Array.isArray(p)) {
      throw new TypeError('snapshot pipe must be an object');
    }
    for (const k of ['id', 'x', 'gapY', 'passed']) {
      if (!(k in p)) {
        throw new TypeError(`snapshot pipe is missing "${k}"`);
      }
    }
    if (!isInteger(p.id) || p.id < 0) {
      throw new RangeError('snapshot pipe id must be a nonnegative integer');
    }
    if (seen.has(p.id)) {
      throw new RangeError('snapshot pipe ids must be unique');
    }
    seen.add(p.id);
    if (typeof p.x !== 'number' || !isFinite(p.x)) {
      throw new RangeError('snapshot pipe x must be finite');
    }
    if (typeof p.gapY !== 'number' || !isFinite(p.gapY)) {
      throw new RangeError('snapshot pipe gapY must be finite');
    }
    if (typeof p.passed !== 'boolean') {
      throw new TypeError('snapshot pipe passed must be a boolean');
    }
  }
  // Everything validated: reject happened atomically; hand back a detached copy.
  const state = structuredClone(snapshot);
  state.config = validateConfig(state.config);
  return state;
}

// Pure fixed-step tick; never mutates state or input.
export function tick(state, input = {}) {
  if (state.phase !== 'running') {
    return cloneState(state);
  }
  const cfg = state.config;
  const s = cloneState(state);
  s.frame += 1;

  if (input && input.flap === true) {
    s.bird.vy = cfg.flapVelocity;
  }
  s.bird.vy += cfg.gravity;
  s.bird.y += s.bird.vy;

  for (const p of s.pipes) {
    p.x -= cfg.pipeSpeed;
  }

  if (s.frame % cfg.spawnEvery === 0) {
    const rng = createRng(s.rngState);
    const span = cfg.height - cfg.groundHeight - cfg.gapHeight - 40 + 1;
    const gapY = 20 + Math.floor(rng.next() * span);
    s.pipes.push({id: s.frame, x: cfg.width, gapY, passed: false});
    s.rngState = rng.snapshot();
  }

  let hit = false;
  for (const p of s.pipes) {
    const right = p.x + cfg.pipeWidth;
    if (!p.passed && right < s.bird.x - cfg.birdRadius) {
      p.passed = true;
      s.score += 1;
    }
    if (right < 0) {
      p.remove = true;
    }
    if (!hit && s.bird.x + cfg.birdRadius >= p.x && s.bird.x - cfg.birdRadius <= right &&
        (s.bird.y - cfg.birdRadius <= p.gapY || s.bird.y + cfg.birdRadius >= p.gapY + cfg.gapHeight)) {
      hit = true;
    }
  }
  s.pipes = s.pipes.filter((p) => !p.remove);
  for (const p of s.pipes) {
    delete p.remove;
  }
  if (hit) {
    s.phase = 'gameover';
    s.score = state.score;
  }
  if (s.bird.y - cfg.birdRadius <= 0 || s.bird.y + cfg.birdRadius >= cfg.height - cfg.groundHeight) {
    s.phase = 'gameover';
    s.score = state.score;
  }
  return s;
}

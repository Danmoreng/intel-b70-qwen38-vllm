// Implementation of specs/simulation.md — pure, deterministic fixed-step game logic.
// No wall clock, no DOM. Every transition returns a detached object and never
// mutates state, input, or config.

import {createRng} from './random.js';
import {validateConfig} from './config.js';

const PHASES = Object.freeze(['ready', 'running', 'paused', 'gameover']);

const isInt = (v) => typeof v === 'number' && Number.isInteger(v);
const isFiniteNum = (v) => typeof v === 'number' && Number.isFinite(v);
const isPlainObject = (v) => typeof v === 'object' && v !== null && !Array.isArray(v);

function initialState(config) {
  const rng = createRng(config.seed);
  return {
    version: 1,
    config,
    phase: 'ready',
    frame: 0,
    score: 0,
    rngState: rng.snapshot(),
    bird: {x: config.birdX, y: (config.height - config.groundHeight) / 2, vy: 0},
    pipes: [],
  };
}

export function createGame(config = {}) {
  return initialState(validateConfig(config));
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
    pipes: state.pipes.map((p) => ({...p})),
  };
}

export function restart(state) {
  return initialState(state.config);
}

export function flap(state) {
  const s = cloneState(state);
  if (s.phase === 'ready') {
    s.phase = 'running';
    s.bird.vy = s.config.flapVelocity;
  } else if (s.phase === 'running') {
    s.bird.vy = s.config.flapVelocity;
  }
  return s; // paused/gameover: detached no-op
}

export function pause(state) {
  const s = cloneState(state);
  if (s.phase === 'running') s.phase = 'paused';
  return s;
}

export function resume(state) {
  const s = cloneState(state);
  if (s.phase === 'paused') s.phase = 'running';
  return s;
}

export function tick(state, input = {}) {
  const s = cloneState(state);
  if (s.phase !== 'running') return s;

  s.frame += 1;
  if (input.flap === true) s.bird.vy = s.config.flapVelocity;
  s.bird.vy += s.config.gravity;
  s.bird.y += s.bird.vy;

  for (const p of s.pipes) p.x -= s.config.pipeSpeed;

  if (s.frame % s.config.spawnEvery === 0) {
    const rng = createRng(s.rngState);
    const span = s.config.height - s.config.groundHeight - s.config.gapHeight - 40 + 1;
    const gapY = 20 + Math.floor(rng.next() * span);
    s.pipes.push({id: s.frame, x: s.config.width, gapY, passed: false});
    s.rngState = rng.snapshot();
  }

  const {bird, config} = s;
  const r = config.birdRadius;
  let collision = bird.y - r <= 0 || bird.y + r >= config.height - config.groundHeight;
  if (!collision) {
    for (const p of s.pipes) {
      const overlap = bird.x + r >= p.x && bird.x - r <= p.x + config.pipeWidth;
      if (overlap && (bird.y - r <= p.gapY || bird.y + r >= p.gapY + config.gapHeight)) {
        collision = true;
        break;
      }
    }
  }
  if (collision) {
    s.phase = 'gameover'; // no points awarded for the colliding frame
    return s;
  }

  for (const p of s.pipes) {
    if (!p.passed && p.x + config.pipeWidth < bird.x - r) {
      p.passed = true;
      s.score += 1;
    }
  }
  s.pipes = s.pipes.filter((p) => p.x + config.pipeWidth >= 0);
  return s;
}

export function restoreState(snapshot) {
  if (!isPlainObject(snapshot) || snapshot.version !== 1) {
    throw new TypeError('restoreState expects a complete state snapshot (version 1)');
  }
  if (!isPlainObject(snapshot.config)) {
    throw new TypeError('snapshot config must be an object');
  }
  const config = validateConfig(snapshot.config);
  if (!Object.prototype.hasOwnProperty.call(snapshot, 'phase') ||
      !PHASES.includes(snapshot.phase)) {
    throw new TypeError('snapshot phase must be a legal phase');
  }
  if (!isInt(snapshot.frame) || snapshot.frame < 0) {
    throw new RangeError('snapshot frame must be a nonnegative integer');
  }
  if (!isInt(snapshot.score) || snapshot.score < 0) {
    throw new RangeError('snapshot score must be a nonnegative integer');
  }
  if (!isInt(snapshot.rngState) || snapshot.rngState === 0 || snapshot.rngState > 0xffffffff) {
    throw new RangeError('snapshot rngState must be a nonzero uint32');
  }
  if (!isPlainObject(snapshot.bird) ||
      !isFiniteNum(snapshot.bird.x) || !isFiniteNum(snapshot.bird.y) || !isFiniteNum(snapshot.bird.vy)) {
    throw new TypeError('snapshot bird must have finite x, y and vy');
  }
  if (!Array.isArray(snapshot.pipes)) {
    throw new TypeError('snapshot pipes must be an array');
  }
  const seenIds = new Set();
  for (const p of snapshot.pipes) {
    if (!isPlainObject(p) || !isInt(p.id) || p.id < 0 ||
        !isFiniteNum(p.x) || !isFiniteNum(p.gapY) ||
        typeof p.passed !== 'boolean' || seenIds.has(p.id)) {
      throw new TypeError('snapshot pipes need unique nonnegative integer ids, finite x/gapY and boolean passed');
    }
    seenIds.add(p.id);
  }
  return {
    version: 1,
    config: {...config},
    phase: snapshot.phase,
    frame: snapshot.frame,
    score: snapshot.score,
    rngState: snapshot.rngState,
    bird: {x: snapshot.bird.x, y: snapshot.bird.y, vy: snapshot.bird.vy},
    pipes: snapshot.pipes.map((p) => ({id: p.id, x: p.x, gapY: p.gapY, passed: p.passed})),
  };
}

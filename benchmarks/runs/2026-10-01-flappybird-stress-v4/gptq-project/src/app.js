// Implementation of specs/app.md — integrates simulation, renderer and input
// into a small playable Flappy Bird. Safely importable in Node: no top-level
// DOM access; everything mounts lazily inside mountApp(root, options).

import {
  createGame, tick, flap as simFlap, pause as simPause,
  resume as simResume, restart as simRestart, cloneState,
} from './simulation.js';
import {validateConfig, DEFAULT_CONFIG} from './config.js';
import {createRenderer} from './renderer.js';
import {createInput, createTicker, normalizePointer} from './input.js';

const PHASE_LABEL = {
  ready: 'Ready — press Space or tap Start',
  running: 'Running',
  paused: 'Paused',
  gameover: 'Game over — press R or tap Restart',
};

export function mountApp(root, options = {}) {
  if (!root || typeof root.appendChild !== 'function') {
    throw new TypeError('mountApp requires a DOM element root');
  }
  const autoStart = options.autoStart !== false;
  const config = options.config ? validateConfig(options.config) : {...DEFAULT_CONFIG};
  const highContrast = options.highContrast === true;
  const reducedMotion = options.reducedMotion === true
    || (typeof window !== 'undefined' && window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches);

  // --- markup (exactly one game canvas, labelled buttons, a11y live region) ---
  root.innerHTML = '';
  const wrap = document.createElement('div');
  wrap.className = 'flappy';
  const canvas = document.createElement('canvas');
  canvas.dataset-testid = 'game-canvas';
  canvas.setAttribute('aria-label', 'Flappy Bird game view');
  const hud = document.createElement('div');
  hud.className = 'flappy-hud';
  const scoreEl = document.createElement('p');
  scoreEl.dataset-testid = 'score';
  const statusEl = document.createElement('p');
  statusEl.dataset-testid = 'status';
  const liveEl = document.createElement('p');
  liveEl.dataset-testid = 'live';
  liveEl.setAttribute('role', 'status');
  liveEl.setAttribute('aria-live', 'polite');
  const controls = document.createElement('div');
  controls.className = 'flappy-controls';
  const startBtn = document.createElement('button');
  startBtn.type = 'button';
  startBtn.dataset-testid = 'start';
  startBtn.textContent = 'Start';
  const pauseBtn = document.createElement('button');
  pauseBtn.type = 'button';
  pauseBtn.dataset-testid = 'pause';
  pauseBtn.textContent = 'Pause';
  const restartBtn = document.createElement('button');
  restartBtn.type = 'button';
  restartBtn.dataset-testid = 'restart';
  restartBtn.textContent = 'Restart';
  hud.append(scoreEl, statusEl);
  controls.append(startBtn, pauseBtn, restartBtn);
  wrap.append(canvas, hud, controls, liveEl);
  root.appendChild(wrap);

  // --- subsystems ---
  let state = createGame(config);
  const input = createInput();
  const renderer = createRenderer(canvas, {highContrast, reducedMotion});
  const ticker = createTicker(onFixedStep, {hz: 60, maxSteps: 5});

  let destroyed = false;
  let rafId = null;
  let flapDebt = 0; // queued flap edges to apply on upcoming ticks

  // --- UI sync (renderer + HUD) ---
  function renderOnce() {
    renderer.render(state, {highContrast, reducedMotion});
    scoreEl.textContent = `Score: ${state.score}`;
    statusEl.textContent = PHASE_LABEL[state.phase];
  }

  function announce(message) {
    liveEl.textContent = message;
  }

  function sizeToViewport() {
    const W = config.width;
    const H = config.height;
    const maxW = Math.min(window.innerWidth || W, W);
    const maxH = (window.innerHeight || H) - 72; // leave room for HUD/controls
    const scale = Math.max(0.2, Math.min(1, maxW / W, maxH / H));
    const dpr = Math.min(4, window.devicePixelRatio || 1);
    renderer.resize(Math.max(120, Math.round(W * scale)), Math.max(180, Math.round(H * scale)), dpr);
  }

  // --- fixed-step advancement (shared by the rAF ticker and step()) ---
  function advanceOne() {
    if (state.phase === 'ready' && flapDebt > 0) {
      flapDebt -= 1; // a queued flap starts the game
      state = simFlap(state);
      announce('Game running');
      return;
    }
    if (state.phase !== 'running') return;
    const doFlap = flapDebt > 0;
    if (doFlap) flapDebt -= 1;
    state = tick(state, {flap: doFlap});
    if (state.phase === 'gameover') {
      announce(`Game over. Score ${state.score}.`);
    }
  }

  function onFixedStep() {
    advanceOne();
    renderOnce();
  }

  function queueActions() {
    const actions = input.consume();
    for (const action of actions) {
      if (action === 'flap') {
        flapDebt += 1;
        if (state.phase === 'paused' || state.phase === 'gameover') flapDebt = 0;
      } else if (action === 'pause') {
        if (state.phase === 'running') api.pause();
        else if (state.phase === 'paused') api.resume(); // KeyP toggles
      } else if (action === 'restart') {
        api.restart();
      }
    }
  }

  // --- public API ---
  const api = {
    start() {
      if (destroyed) throw new Error('app is destroyed');
      if (state.phase === 'running') return;
      if (state.phase === 'gameover') state = simRestart(state);
      if (state.phase === 'paused') state = simResume(state);
      flapDebt = 0; // explicit start absorbs any queued flap edge
      if (state.phase === 'ready') state = simFlap(state);
      ticker.resume();
      announce('Game running');
      renderOnce();
    },

    flap() {
      if (destroyed) throw new Error('app is destroyed');
      if (state.phase === 'paused' || state.phase === 'gameover') return;
      state = simFlap(state);
      if (state.phase === 'running' && state.frame === 0) {
        ticker.resume();
        announce('Game running');
      }
      renderOnce();
    },

    pause() {
      if (destroyed) throw new Error('app is destroyed');
      if (state.phase !== 'running') return;
      state = simPause(state);
      flapDebt = 0;
      ticker.pause();
      announce('Paused');
      renderOnce();
    },

    resume() {
      if (destroyed) throw new Error('app is destroyed');
      if (state.phase !== 'paused') return;
      state = simResume(state);
      ticker.resume();
      announce('Game running');
      renderOnce();
    },

    restart() {
      if (destroyed) throw new Error('app is destroyed');
      state = simRestart(state);
      flapDebt = 0;
      ticker.reset();
      announce('Reset — ready');
      renderOnce();
    },

    step(count = 1) {
      if (destroyed) throw new Error('app is destroyed');
      if (!Number.isInteger(count) || count < 0 || count > 10000) {
        throw new RangeError('step count must be an integer 0..10000');
      }
      queueActions();
      for (let i = 0; i < count; i += 1) {
        advanceOne();
        if (state.phase === 'gameover') break;
      }
      renderOnce();
    },

    getState() {
      return cloneState(state);
    },

    destroy() {
      if (destroyed) return; // idempotent
      destroyed = true;
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      input.dispose();
      renderer.destroy();
      window.removeEventListener('keydown', onKeyDown, false);
      window.removeEventListener('keyup', onKeyUp, false);
      document.removeEventListener('visibilitychange', onVisibility, false);
      window.removeEventListener('resize', onResize, false);
      canvas.removeEventListener('pointerdown', onPointerDown, false);
      startBtn.removeEventListener('click', onStartClick, false);
      pauseBtn.removeEventListener('click', onPauseClick, false);
      restartBtn.removeEventListener('click', onRestartClick, false);
    },
  };

  // --- event wiring ---
  function typingTarget(target) {
    return !!(target && target.closest && (target.closest('input, textarea') !== null));
  }

  function onKeyDown(event) {
    if (destroyed) return;
    if (typingTarget(event.target)) return; // ignore while input/textarea focused
    if (event.repeat) return; // edge-triggered; also handled inside createInput
    if (input.handle({code: event.code, repeat: event.repeat, type: event.type}) !== true) {
      return;
    }
    if (event.code === 'Space') event.preventDefault();
    queueActions();
    if (flapDebt > 0 && state.phase === 'running') {
      // interactive flap: apply immediately for responsiveness
      const doFlap = flapDebt > 0;
      if (doFlap) flapDebt -= 1;
      state = tick(state, {flap: doFlap});
      renderOnce();
    } else if (state.phase === 'ready') {
      state = simFlap(state);
      flapDebt = Math.max(0, flapDebt - 1);
      announce('Game running');
      renderOnce();
    }
  }

  function onKeyUp(event) {
    if (destroyed) return;
    if (typingTarget(event.target)) return;
    input.release({code: event.code, type: 'keyup'});
  }

  function onVisibility() {
    if (destroyed) return;
    if (document.hidden && state.phase === 'running') api.pause();
  }

  function onResize() {
    if (destroyed) return;
    sizeToViewport();
    renderOnce();
  }

  function onPointerDown(event) {
    if (destroyed) return;
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const logical = normalizePointer(event, rect, config.width, config.height);
    renderer.render(state, {highContrast, reducedMotion, pointer: {x: logical.x, y: logical.y}});
    api.flap();
  }

  function onStartClick() { if (!destroyed) api.start(); }
  function onPauseClick() {
    if (destroyed) return;
    if (state.phase === 'running') api.pause();
    else if (state.phase === 'paused') api.resume();
  }
  function onRestartClick() { if (!destroyed) api.restart(); }

  window.addEventListener('keydown', onKeyDown, false);
  window.addEventListener('keyup', onKeyUp, false);
  document.addEventListener('visibilitychange', onVisibility, false);
  window.addEventListener('resize', onResize, false);
  canvas.addEventListener('pointerdown', onPointerDown, false);
  startBtn.addEventListener('click', onStartClick, false);
  pauseBtn.addEventListener('click', onPauseClick, false);
  restartBtn.addEventListener('click', onRestartClick, false);

  // --- boot ---
  sizeToViewport();
  renderOnce();
  if (autoStart) {
    const frame = (timestamp) => {
      if (destroyed) return;
      const n = ticker.update(timestamp);
      if (n > 0) queueActions();
      rafId = requestAnimationFrame(frame);
    };
    rafId = requestAnimationFrame(frame);
  }

  return api;
}

// Public API contract: specs/app.md
import {DEFAULT_CONFIG} from './config.js';
import {
  createGame,
  tick,
  flap as simFlap,
  pause as simPause,
  resume as simResume,
  restart as simRestart,
  cloneState
} from './simulation.js';
import {createRenderer} from './renderer.js';
import {createInput, createTicker} from './input.js';

const STATUS_TEXT = {
  ready: 'Ready',
  running: 'Running',
  paused: 'Paused',
  gameover: 'Game over'
};

export function mountApp(root, options = {}) {
  if (typeof root !== 'object' || root === null || typeof root.appendChild !== 'function') {
    throw new TypeError('root must be a DOM element');
  }
  const doc = root.ownerDocument;
  const autoStart = options.autoStart !== false;

  root.textContent = '';
  const layout = doc.createElement('div');
  layout.className = 'flappy';

  const canvas = doc.createElement('canvas');
  canvas.setAttribute('data-testid', 'game-canvas');
  canvas.setAttribute('aria-label', 'Flappy bird game canvas');

  const scoreEl = doc.createElement('div');
  scoreEl.setAttribute('data-testid', 'score');
  scoreEl.className = 'flappy-score';
  scoreEl.textContent = '0';

  const statusEl = doc.createElement('div');
  statusEl.setAttribute('data-testid', 'status');
  statusEl.className = 'flappy-status';

  const liveEl = doc.createElement('div');
  liveEl.setAttribute('data-testid', 'live');
  liveEl.setAttribute('role', 'status');
  liveEl.setAttribute('aria-live', 'polite');
  liveEl.className = 'flappy-live';

  const startBtn = doc.createElement('button');
  startBtn.setAttribute('data-testid', 'start');
  startBtn.type = 'button';
  startBtn.textContent = 'Start';

  const pauseBtn = doc.createElement('button');
  pauseBtn.setAttribute('data-testid', 'pause');
  pauseBtn.type = 'button';
  pauseBtn.textContent = 'Pause';

  const restartBtn = doc.createElement('button');
  restartBtn.setAttribute('data-testid', 'restart');
  restartBtn.type = 'button';
  restartBtn.textContent = 'Restart';

  layout.append(canvas, scoreEl, statusEl, liveEl, startBtn, pauseBtn, restartBtn);
  root.appendChild(layout);

  let renderer;
  try {
    renderer = createRenderer(canvas);
  } catch (error) {
    throw error;
  }

  const input = createInput();
  const media =
    typeof window !== 'undefined' && typeof window.matchMedia === 'function'
      ? window.matchMedia
      : null;
  const reducedMotion =
    media !== null && media('(prefers-reduced-motion: reduce)').matches === true;
  const highContrast =
    media !== null && media('(prefers-contrast: high)').matches === true;

  let state = createGame(DEFAULT_CONFIG);
  let destroyed = false;
  let rafId = null;
  let hiddenPaused = false;
  let lastScore = 0;

  const ticker = createTicker(() => {
    doFrame();
  }, {hz: 60, maxSteps: 5});

  function renderFrame() {
    if (destroyed || renderer.info().destroyed) {
      return;
    }
    renderer.render(state, {highContrast, reducedMotion});
    scoreEl.textContent = String(state.score);
    statusEl.textContent = STATUS_TEXT[state.phase] || state.phase;
    if (state.score !== lastScore) {
      lastScore = state.score;
      liveEl.textContent = `Score ${state.score}`;
    }
    if (state.phase === 'gameover') {
      liveEl.textContent = `Game over. Score ${state.score}.`;
    } else if (state.phase === 'ready' && lastScore === 0) {
      liveEl.textContent = 'Ready. Press start or tap to begin.';
    }
  }

  function doFrame() {
    if (destroyed) {
      return;
    }
    const actions = input.consume();
    for (const action of actions) {
      if (action === 'restart') {
        state = simRestart(state);
      } else if (action === 'pause') {
        state =
          state.phase === 'running' ? simPause(state) : state.phase === 'paused' ? simResume(state) : state;
      } else if (action === 'flap' && state.phase === 'ready') {
        state = simFlap(state);
      }
    }
    let flapFlag = actions.includes('flap');
    if (state.phase === 'running') {
      state = tick(state, {flap: flapFlag});
      flapFlag = false;
    }
    renderFrame();
  }

  function layoutCanvas() {
    if (destroyed || renderer.info().destroyed) {
      return;
    }
    const dpr = Math.min(
      typeof window !== 'undefined' && typeof window.devicePixelRatio === 'number'
        ? window.devicePixelRatio
        : 1,
      4
    );
    const vw = root.clientWidth || (typeof window !== 'undefined' ? window.innerWidth : 0) || 480;
    const vh =
      root.clientHeight ||
      (typeof document !== 'undefined' && doc.body ? doc.body.clientHeight : 0) ||
      (typeof window !== 'undefined' ? window.innerHeight : 0) ||
      720;
    const scale = Math.max(
      0.2,
      Math.min(vw / state.config.width, vh / state.config.height) * 0.96
    );
    renderer.resize(
      state.config.width * scale,
      state.config.height * scale,
      dpr
    );
    renderFrame();
  }

  function applyFlap() {
    state = simFlap(state);
    renderFrame();
  }

  function startLoop() {
    if (rafId !== null || destroyed || !autoStart) {
      return;
    }
    const frame = (ts) => {
      if (destroyed) {
        return;
      }
      rafId = requestAnimationFrame(frame);
      ticker.update(ts);
    };
    rafId = requestAnimationFrame(frame);
  }

  const api = {
    start() {
      if (destroyed) {
        throw new Error('app is destroyed');
      }
      if (state.phase === 'ready') {
        state = simFlap(state);
      } else if (state.phase === 'paused') {
        state = simResume(state);
      } else if (state.phase === 'gameover') {
        state = simFlap(simRestart(state));
      }
      renderFrame();
      startLoop();
    },
    step(count = 1) {
      if (destroyed) {
        throw new Error('app is destroyed');
      }
      if (!Number.isInteger(count) || count < 0 || count > 10000) {
        throw new TypeError('step count must be an integer from 0 to 10000');
      }
      for (let i = 0; i < count; i += 1) {
        doFrame();
      }
    },
    flap() {
      if (destroyed) {
        throw new Error('app is destroyed');
      }
      applyFlap();
    },
    pause() {
      if (destroyed) {
        throw new Error('app is destroyed');
      }
      state = simPause(state);
      renderFrame();
    },
    resume() {
      if (destroyed) {
        throw new Error('app is destroyed');
      }
      state = simResume(state);
      renderFrame();
    },
    restart() {
      if (destroyed) {
        throw new Error('app is destroyed');
      }
      state = simRestart(state);
      lastScore = 0;
      renderFrame();
    },
    getState() {
      return cloneState(state);
    },
    destroy() {
      if (destroyed) {
        return;
      }
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      input.dispose();
      try {
        renderer.destroy();
      } catch (error) {
        // renderer may already be gone if the context was hard-removed
      }
      doc.removeEventListener('keydown', onKeydown, true);
      doc.removeEventListener('keyup', onKeyup, true);
      doc.removeEventListener('visibilitychange', onVisibility, false);
      window.removeEventListener('resize', onResize, false);
      canvas.removeEventListener('pointerdown', onPointer, false);
      startBtn.removeEventListener('click', onStartBtn, false);
      pauseBtn.removeEventListener('click', onPauseBtn, false);
      restartBtn.removeEventListener('click', onRestartBtn, false);
      layout.remove();
      destroyed = true;
    }
  };

  function isTypingTarget() {
    const active = doc.activeElement;
    if (!active || typeof active.tagName !== 'string') {
      return false;
    }
    return active.tagName === 'INPUT' || active.tagName === 'TEXTAREA';
  }

  function onKeydown(event) {
    if (destroyed || event.repeat || isTypingTarget()) {
      return;
    }
    if (event.code === 'Space' || event.code === 'ArrowUp') {
      event.preventDefault();
      input.handle({code: event.code, repeat: false, type: 'keydown'});
      const actions = input.consume();
      if (actions.includes('flap')) {
        if (state.phase === 'ready' || state.phase === 'running') {
          state = simFlap(state);
        }
      }
      renderFrame();
    } else if (event.code === 'KeyP') {
      state =
        state.phase === 'running' ? simPause(state) : state.phase === 'paused' ? simResume(state) : state;
      renderFrame();
    } else if (event.code === 'KeyR') {
      state = simRestart(state);
      lastScore = 0;
      renderFrame();
    }
  }

  function onKeyup(event) {
    if (destroyed) {
      return;
    }
    input.release({code: event.code, type: 'keyup'});
  }

  function onPointer(event) {
    if (destroyed) {
      return;
    }
    event.preventDefault();
    applyFlap();
  }

  function onVisibility() {
    if (destroyed) {
      return;
    }
    if (doc.visibilityState === 'hidden') {
      if (state.phase === 'running') {
        state = simPause(state);
        hiddenPaused = true;
        renderFrame();
      }
    } else if (hiddenPaused) {
      hiddenPaused = false;
      state = simResume(state);
      renderFrame();
    }
  }

  function onResize() {
    layoutCanvas();
  }

  function onStartBtn() {
    api.start();
  }

  function onPauseBtn() {
    if (destroyed) {
      return;
    }
    if (state.phase === 'running') {
      api.pause();
    } else if (state.phase === 'paused') {
      api.resume();
    }
  }

  function onRestartBtn() {
    if (!destroyed) {
      api.restart();
    }
  }

  doc.addEventListener('keydown', onKeydown, true);
  doc.addEventListener('keyup', onKeyup, true);
  doc.addEventListener('visibilitychange', onVisibility, false);
  window.addEventListener('resize', onResize, false);
  canvas.addEventListener('pointerdown', onPointer, false);
  startBtn.addEventListener('click', onStartBtn, false);
  pauseBtn.addEventListener('click', onPauseBtn, false);
  restartBtn.addEventListener('click', onRestartBtn, false);

  layoutCanvas();
  renderFrame();
  startLoop();

  return api;
}

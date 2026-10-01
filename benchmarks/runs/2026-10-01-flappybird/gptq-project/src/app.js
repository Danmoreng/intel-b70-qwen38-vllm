// Public API contract: specs/app.md
// Integrates simulation, renderer and input; safely importable in Node.
import {validateConfig} from './config.js';
import {createGame, tick, flap as simFlap, pause as simPause, resume as simResume, restart as simRestart, cloneState} from './simulation.js';
import {createRenderer} from './renderer.js';
import {createInput, createTicker} from './input.js';
import {createScoreStore} from './scores.js';

const PHASE_LABEL = {ready: 'Ready', running: 'Running', paused: 'Paused', gameover: 'Game over'};

const PHASE_MESSAGE = {
  ready: 'Ready. Press Start, Space or tap the canvas to flap.',
  running: 'Running.',
  paused: 'Paused. Press P or the Pause button to resume.',
  gameover: 'Game over. Press R or Restart to play again.'
};

export function mountApp(root, options = {}) {
  if (root === null || root === undefined || typeof root.appendChild !== 'function' ||
      typeof root.addEventListener !== 'function') {
    throw new TypeError('mountApp root must be a DOM element');
  }
  const autoStart = options.autoStart !== false;
  const highContrast = options.highContrast === true;
  const reducedMotion = options.reducedMotion === true ||
    (options.reducedMotion !== false && typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const config = validateConfig(options.config || {});

  // ---- state -------------------------------------------------------------
  let state = createGame(config);
  let lastPhase = 'ready';
  let destroyed = false;
  let rafId = null;
  let scores = [];

  // ---- DOM ---------------------------------------------------------------
  root.innerHTML = '';
  const game = document.createElement('div');
  game.className = 'game';

  const canvas = document.createElement('canvas');
  canvas.setAttribute('data-testid', 'game-canvas');
  canvas.setAttribute('aria-label', 'Flappy Bird game view');

  const hud = document.createElement('div');
  hud.className = 'hud';
  const scoreEl = document.createElement('span');
  scoreEl.setAttribute('data-testid', 'score');
  scoreEl.className = 'score';
  const statusEl = document.createElement('span');
  statusEl.setAttribute('data-testid', 'status');
  statusEl.className = 'status';
  hud.append(scoreEl, statusEl);

  const liveEl = document.createElement('p');
  liveEl.setAttribute('data-testid', 'live');
  liveEl.setAttribute('role', 'status');
  liveEl.setAttribute('aria-live', 'polite');
  liveEl.className = 'live';

  const controls = document.createElement('div');
  controls.className = 'controls';
  const startBtn = document.createElement('button');
  startBtn.setAttribute('data-testid', 'start');
  startBtn.type = 'button';
  startBtn.textContent = 'Start';
  const pauseBtn = document.createElement('button');
  pauseBtn.setAttribute('data-testid', 'pause');
  pauseBtn.type = 'button';
  pauseBtn.textContent = 'Pause';
  const restartBtn = document.createElement('button');
  restartBtn.setAttribute('data-testid', 'restart');
  restartBtn.type = 'button';
  restartBtn.textContent = 'Restart';
  controls.append(startBtn, pauseBtn, restartBtn);

  const settings = document.createElement('section');
  settings.className = 'settings';
  settings.setAttribute('aria-label', 'Game settings');
  const settingsForm = document.createElement('div');
  settingsForm.className = 'settings-grid';
  const SETTING_META = {
    seed: ['Seed', '0', '1'],
    gravity: ['Gravity', '0.01', '0.01'],
    gap: ['Gap height', '1', '60']
  };
  const settingInputs = {};
  for (const field of ['seed', 'gravity', 'gap']) {
    const [text, step, min] = SETTING_META[field];
    const label = document.createElement('label');
    const inputEl = document.createElement('input');
    inputEl.type = 'number';
    inputEl.id = `settings-${field}`;
    inputEl.setAttribute('data-testid', `settings-${field}`);
    inputEl.step = step;
    inputEl.min = min;
    inputEl.value = String(field === 'seed' ? config.seed : field === 'gravity' ? config.gravity : config.gapHeight);
    settingInputs[field] = inputEl;
    label.append(text, ' ', inputEl);
    settingsForm.append(label);
  }
  const applyBtn = document.createElement('button');
  applyBtn.type = 'button';
  applyBtn.setAttribute('data-testid', 'settings-apply');
  applyBtn.textContent = 'Apply settings';
  const settingsErr = document.createElement('p');
  settingsErr.setAttribute('data-testid', 'settings-error');
  settingsErr.className = 'settings-error';
  settingsErr.setAttribute('role', 'alert');
  settingsErr.setAttribute('aria-live', 'assertive');
  settingsForm.append(applyBtn, settingsErr);
  settings.append(settingsForm);

  const nameForm = document.createElement('p');
  nameForm.className = 'player-name';
  const nameLabel = document.createElement('label');
  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.id = 'player-name';
  nameInput.maxLength = 24;
  nameInput.value = 'Player';
  nameLabel.append('Player name ', nameInput);
  nameForm.append(nameLabel);

  const scoresSec = document.createElement('section');
  scoresSec.className = 'scores';
  scoresSec.setAttribute('aria-label', 'High scores');
  const clearBtn = document.createElement('button');
  clearBtn.type = 'button';
  clearBtn.setAttribute('data-testid', 'scores-clear');
  clearBtn.textContent = 'Clear scores';
  const scoreList = document.createElement('ul');
  scoreList.className = 'score-list';
  scoresSec.append(clearBtn, scoreList);

  game.append(canvas, hud, liveEl, controls, settings, nameForm, scoresSec);
  root.appendChild(game);

  // ---- subsystems ----------------------------------------------------------
  const renderer = createRenderer(canvas, {highContrast});
  const input = createInput();
  const ticker = createTicker(tickerStep, {hz: 60, maxSteps: 5});
  let memoryStorage = {};
  const scoresStorage = typeof localStorage !== 'undefined'
    ? {
        getItem: (k) => localStorage.getItem(k),
        setItem: (k, v) => localStorage.setItem(k, v),
        removeItem: (k) => localStorage.removeItem(k)
      }
    : {
        getItem: (k) => (k in memoryStorage ? memoryStorage[k] : null),
        setItem: (k, v) => {
          memoryStorage[k] = String(v);
        },
        removeItem: (k) => {
          delete memoryStorage[k];
        }
      };
  const scoreStore = createScoreStore(scoresStorage);
  scores = scoreStore.list();

  // ---- sizing (responsive, DPR capped at 4) -------------------------------
  function fit() {
    if (destroyed) {
      return;
    }
    const dpr = Math.min(typeof window.devicePixelRatio === 'number' ? window.devicePixelRatio : 1, 4);
    renderer.resize(config.width, config.height, dpr);
    const container = root.clientWidth || (window.innerWidth || config.width);
    const cssW = Math.max(200, Math.min(config.width, container - 24));
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${Math.round(cssW * config.height / config.width)}px`;
  }

  // ---- UI sync -------------------------------------------------------------
  function renderScores() {
    scoreList.innerHTML = '';
    for (const entry of scores) {
      const li = document.createElement('li');
      li.setAttribute('data-testid', 'score-row');
      const nameSpan = document.createElement('span');
      nameSpan.className = 'score-name';
      nameSpan.textContent = entry.name;
      const valueSpan = document.createElement('span');
      valueSpan.className = 'score-value';
      valueSpan.textContent = String(entry.score);
      li.append(nameSpan, valueSpan);
      scoreList.append(li);
    }
  }

  // Exactly one record per running -> gameover transition.
  function recordFinalScore() {
    const name = nameInput.value.trim().slice(0, 24) || 'Player';
    scores = scoreStore.record({name, score: state.score, frames: state.frame, seed: state.config.seed});
    renderScores();
  }

  // Apply the three settings atomically; any validation failure keeps the
  // whole current config and game state untouched.
  function applySettings() {
    guard();
    const read = (el) => {
      const text = el.value.trim();
      return text === '' ? Number.NaN : Number(text);
    };
    const seed = read(settingInputs.seed);
    const gravity = read(settingInputs.gravity);
    const gap = read(settingInputs.gap);
    let validated;
    try {
      validated = validateConfig({...config, seed, gravity, gapHeight: gap});
    } catch (err) {
      settingsErr.textContent = `Settings not applied: ${err.message}`;
      return;
    }
    settingsErr.textContent = '';
    config.seed = validated.seed;
    config.gravity = validated.gravity;
    config.gapHeight = validated.gapHeight;
    state = createGame(config);
    lastPhase = state.phase;
    fit();
    sync();
  }

  function clearScores() {
    guard();
    scoreStore.clear();
    scores = [];
    renderScores();
  }

  function sync() {
    if (destroyed) {
      return;
    }
    if (lastPhase === 'running' && state.phase === 'gameover') {
      recordFinalScore();
    }
    lastPhase = state.phase;
    scoreEl.textContent = String(state.score);
    statusEl.textContent = PHASE_LABEL[state.phase];
    let message = PHASE_MESSAGE[state.phase];
    if (state.phase === 'running') {
      message = `Running. Score ${state.score}.`;
    } else if (state.phase === 'gameover') {
      message = `Game over. Final score ${state.score}. Press R or Restart to play again.`;
    }
    liveEl.textContent = message;
    startBtn.disabled = state.phase === 'running';
    pauseBtn.textContent = state.phase === 'paused' ? 'Resume' : 'Pause';
    pauseBtn.disabled = state.phase !== 'running' && state.phase !== 'paused';
    renderer.render(state, {highContrast, reducedMotion});
  }

  // ---- actions ---------------------------------------------------------------
  function guard() {
    if (destroyed) {
      throw new Error('app is destroyed');
    }
  }

  function start() {
    guard();
    if (state.phase === 'ready') {
      state = simFlap(state);
    } else if (state.phase === 'paused') {
      state = simResume(state);
      if (autoStart) {
        ticker.resume();
      }
    }
    sync();
  }

  function flap() {
    guard();
    state = simFlap(state);
    sync();
  }

  function pause() {
    guard();
    if (state.phase !== 'running') {
      return;
    }
    state = simPause(state);
    if (autoStart) {
      ticker.pause();
    }
    sync();
  }

  function resume() {
    guard();
    if (state.phase !== 'paused') {
      return;
    }
    state = simResume(state);
    if (autoStart) {
      ticker.resume();
    }
    sync();
  }

  function restart() {
    guard();
    state = simRestart(state);
    if (autoStart) {
      ticker.resume();
    }
    sync();
  }

  function togglePause() {
    if (state.phase === 'running') {
      pause();
    } else if (state.phase === 'paused') {
      resume();
    }
  }

  // Edge-triggered actions from the input queue (keyboard edges).
  function applyActions(actions) {
    for (const action of actions) {
      if (action === 'flap') {
        state = simFlap(state);
      } else if (action === 'pause') {
        togglePause();
      } else if (action === 'restart') {
        state = simRestart(state);
        if (autoStart) {
          ticker.resume();
        }
      }
    }
  }

  function step(count = 1) {
    guard();
    if (typeof count !== 'number' || !Number.isInteger(count) || count < 0 || count > 10000) {
      throw new RangeError('step count must be an integer within 0..10000');
    }
    applyActions(input.consume());
    if (state.phase === 'running') {
      for (let i = 0; i < count; i++) {
        state = tick(state, {});
        if (state.phase !== 'running') {
          break;
        }
      }
      if (autoStart) {
        ticker.reset();
      }
    }
    sync();
  }

  // One fixed-dt step driven by the ticker inside the rAF loop.
  function tickerStep() {
    if (destroyed) {
      return;
    }
    applyActions(input.consume());
    if (state.phase !== 'running') {
      sync();
      return;
    }
    state = tick(state, {});
    sync();
  }

  // ---- event wiring ------------------------------------------------------------
  function isFormField(el) {
    if (el === null || el === undefined) {
      return false;
    }
    const tag = el.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
  }

  function onKeydown(e) {
    if (destroyed || isFormField(document.activeElement)) {
      return;
    }
    // Repeat-held keydowns return true from handle() without queueing a new action.
    if (input.handle({code: e.code, repeat: e.repeat, type: 'keydown'})) {
      e.preventDefault();
      applyActions(input.consume());
    }
  }
  function onKeyup(e) {
    if (!destroyed) {
      input.release({code: e.code});
    }
  }
  function onPointer(e) {
    if (destroyed) {
      return;
    }
    if (e.type === 'pointerdown') {
      e.preventDefault();
      flap();
    }
  }
  function onVisibility() {
    if (destroyed) {
      return;
    }
    if (document.hidden && state.phase === 'running') {
      pause();
    }
  }
  function onResize() {
    if (!destroyed) {
      fit();
      sync();
    }
  }

  window.addEventListener('keydown', onKeydown);
  window.addEventListener('keyup', onKeyup);
  canvas.addEventListener('pointerdown', onPointer);
  startBtn.addEventListener('click', start);
  pauseBtn.addEventListener('click', togglePause);
  restartBtn.addEventListener('click', restart);
  applyBtn.addEventListener('click', applySettings);
  clearBtn.addEventListener('click', clearScores);
  document.addEventListener('visibilitychange', onVisibility);
  window.addEventListener('resize', onResize);

  renderScores();
  fit();
  sync();

  function frame(ts) {
    if (destroyed) {
      return;
    }
    rafId = window.requestAnimationFrame(frame);
    try {
      ticker.update(ts);
    } catch {
      // invalid timestamps are rejected; the loop keeps a fresh base
      ticker.reset();
    }
  }
  if (autoStart) {
    rafId = window.requestAnimationFrame(frame);
  }

  // ---- public API -------------------------------------------------------------
  function destroy() {
    if (destroyed) {
      return;
    }
    destroyed = true;
    if (rafId !== null) {
      window.cancelAnimationFrame(rafId);
      rafId = null;
    }
    window.removeEventListener('keydown', onKeydown);
    window.removeEventListener('keyup', onKeyup);
    window.removeEventListener('resize', onResize);
    document.removeEventListener('visibilitychange', onVisibility);
    canvas.removeEventListener('pointerdown', onPointer);
    input.dispose();
    renderer.destroy();
  }

  return {
    start,
    step,
    flap,
    pause,
    resume,
    restart,
    getState() {
      return cloneState(state);
    },
    destroy
  };
}

// Public API contract: specs/app.md
import {validateConfig} from './config.js';
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
import {createScoreStore} from './scores.js';

const PHASE_LABELS = {
  ready: 'Ready',
  running: 'Running',
  paused: 'Paused',
  gameover: 'Game over'
};

function isEditableTarget(target) {
  let el = target;
  while (el && typeof el.tagName === 'string') {
    const tag = el.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
      return true;
    }
    if (el.isContentEditable === true) {
      return true;
    }
    el = el.parentNode;
  }
  return false;
}

export function mountApp(root, options = {}) {
  if (root === null || typeof root !== 'object' || typeof root.appendChild !== 'function') {
    throw new Error('mountApp expects a DOM element');
  }
  const opts = options !== null && typeof options === 'object' ? options : {};
  const autoStart = opts.autoStart !== false;
  let config = validateConfig(
    opts.config !== null && typeof opts.config === 'object' ? opts.config : {}
  );

  // ---- Markup: exactly the required test-id elements.
  root.textContent = '';
  const wrap = document.createElement('div');
  wrap.className = 'flappy-app';

  const canvas = document.createElement('canvas');
  canvas.setAttribute('data-testid', 'game-canvas');

  function makeButton(testId, label) {
    const btn = document.createElement('button');
    btn.setAttribute('data-testid', testId);
    btn.textContent = label;
    btn.type = 'button';
    return btn;
  }
  const startBtn = makeButton('start', 'Start');
  const pauseBtn = makeButton('pause', 'Pause');
  const restartBtn = makeButton('restart', 'Restart');

  const scoreEl = document.createElement('div');
  scoreEl.setAttribute('data-testid', 'score');
  scoreEl.className = 'flappy-score';
  scoreEl.setAttribute('aria-label', 'Score');
  scoreEl.textContent = '0';

  const statusEl = document.createElement('div');
  statusEl.setAttribute('data-testid', 'status');
  statusEl.className = 'flappy-status';
  statusEl.setAttribute('aria-label', 'Game status');
  statusEl.textContent = PHASE_LABELS.ready;

  const liveEl = document.createElement('div');
  liveEl.setAttribute('data-testid', 'live');
  liveEl.className = 'flappy-live';
  liveEl.setAttribute('role', 'status');
  liveEl.setAttribute('aria-live', 'polite');
  liveEl.textContent = '';

  function makeLabeledInput(testId, label, value, type = 'number') {
    const labelEl = document.createElement('label');
    labelEl.className = 'flappy-field';
    labelEl.textContent = label;
    const input = document.createElement('input');
    input.setAttribute('data-testid', testId);
    input.type = type;
    input.value = String(value);
    if (type === 'text') {
      input.maxLength = 24; // scores contract: names up to 24 characters
    }
    labelEl.appendChild(input);
    return labelEl;
  }

  const settingsSection = document.createElement('section');
  settingsSection.className = 'flappy-settings';
  const settingsHeading = document.createElement('h2');
  settingsHeading.textContent = 'Settings';
  const seedField = makeLabeledInput('settings-seed', 'Seed', config.seed);
  const gravityField = makeLabeledInput('settings-gravity', 'Gravity', config.gravity);
  const gapField = makeLabeledInput('settings-gap', 'Gap height', config.gapHeight);
  const applyBtn = document.createElement('button');
  applyBtn.setAttribute('data-testid', 'settings-apply');
  applyBtn.type = 'button';
  applyBtn.textContent = 'Apply';
  const settingsError = document.createElement('div');
  settingsError.setAttribute('data-testid', 'settings-error');
  settingsError.className = 'flappy-error';
  settingsError.setAttribute('role', 'alert');
  settingsError.textContent = '';
  settingsSection.appendChild(settingsHeading);
  settingsSection.appendChild(seedField);
  settingsSection.appendChild(gravityField);
  settingsSection.appendChild(gapField);
  settingsSection.appendChild(applyBtn);
  settingsSection.appendChild(settingsError);

  const playerSection = document.createElement('section');
  playerSection.className = 'flappy-player';
  playerSection.appendChild(makeLabeledInput('player-name', 'Player name', 'Player', 'text'));

  const scoresSection = document.createElement('section');
  scoresSection.className = 'flappy-scores';
  const scoresHeading = document.createElement('h2');
  scoresHeading.textContent = 'High scores';
  const scoreRows = document.createElement('div');
  scoreRows.className = 'flappy-score-rows';
  const clearBtn = document.createElement('button');
  clearBtn.setAttribute('data-testid', 'scores-clear');
  clearBtn.type = 'button';
  clearBtn.textContent = 'Clear scores';
  scoresSection.appendChild(scoresHeading);
  scoresSection.appendChild(scoreRows);
  scoresSection.appendChild(clearBtn);

  wrap.appendChild(canvas);
  wrap.appendChild(scoreEl);
  wrap.appendChild(statusEl);
  const bar = document.createElement('div');
  bar.className = 'flappy-controls';
  bar.appendChild(startBtn);
  bar.appendChild(pauseBtn);
  bar.appendChild(restartBtn);
  wrap.appendChild(bar);
  wrap.appendChild(settingsSection);
  wrap.appendChild(playerSection);
  wrap.appendChild(scoresSection);
  wrap.appendChild(liveEl);
  root.appendChild(wrap);

  const renderer = createRenderer(canvas);
  let state = createGame(config);
  const input = createInput();
  let destroyed = false;

  // ---- Settings inputs / score store wiring.
  const seedInput = seedField.querySelector('input');
  const gravityInput = gravityField.querySelector('input');
  const gapInput = gapField.querySelector('input');
  const nameInput = playerSection.querySelector('input');

  let scoreStore = null;
  try {
    scoreStore = createScoreStore(window.localStorage);
  } catch (err) {
    scoreStore = null;
  }

  function renderScores() {
    if (destroyed || !scoreStore) {
      return;
    }
    let rows;
    try {
      rows = scoreStore.list();
    } catch (err) {
      return;
    }
    scoreRows.textContent = '';
    for (const entry of rows) {
      const row = document.createElement('div');
      row.setAttribute('data-testid', 'score-row');
      row.className = 'flappy-score-row';
      const nameSpan = document.createElement('span');
      nameSpan.className = 'flappy-score-row-name';
      nameSpan.textContent = entry.name;
      const scoreSpan = document.createElement('span');
      scoreSpan.className = 'flappy-score-row-score';
      scoreSpan.textContent = String(entry.score);
      row.appendChild(nameSpan);
      row.appendChild(scoreSpan);
      scoreRows.appendChild(row);
    }
  }

  function recordScore() {
    if (destroyed || !scoreStore) {
      return;
    }
    let rawName = '';
    try {
      rawName = nameInput.value;
    } catch (err) {
      rawName = '';
    }
    const name = (typeof rawName === 'string' ? rawName : '').trim() || 'Player';
    try {
      scoreStore.record({
        name,
        score: state.score,
        frames: state.frame,
        seed: state.config.seed
      });
    } catch (err) {
      return; // invalid name or storage failure: never break the game
    }
    renderScores();
  }

  function reconfigure(nextConfig) {
    config = nextConfig;
    applySize();
  }

  function applySettings() {
    guard();
    settingsError.textContent = '';
    const candidate = {
      seed: seedInput.value === '' ? undefined : Number(seedInput.value),
      gravity: gravityInput.value === '' ? undefined : Number(gravityInput.value),
      gapHeight: gapInput.value === '' ? undefined : Number(gapInput.value)
    };
    let next;
    try {
      next = validateConfig(Object.assign({}, state.config, candidate));
    } catch (err) {
      settingsError.textContent =
        'Invalid settings: ' +
        (err && typeof err.message === 'string'
          ? err.message
          : 'check the seed, gravity and gap values');
      return; // game config and state are preserved
    }
    reconfigure(next);
    state = createGame(next);
    seedInput.value = String(next.seed);
    gravityInput.value = String(next.gravity);
    gapInput.value = String(next.gapHeight);
    renderAndUI();
  }
  let rafId = null;
  let lastLiveKey = '';

  const reducedMotionQuery =
    typeof window.matchMedia === 'function'
      ? window.matchMedia('(prefers-reduced-motion: reduce)')
      : null;

  function renderAndUI() {
    if (destroyed) {
      return;
    }
    renderer.render(state, {
      reducedMotion: !!(reducedMotionQuery && reducedMotionQuery.matches)
    });
    scoreEl.textContent = String(state.score);
    statusEl.textContent = PHASE_LABELS[state.phase];
    const key = state.phase + '|' + state.score;
    if (key !== lastLiveKey) {
      lastLiveKey = key;
      liveEl.textContent =
        state.phase === 'gameover'
          ? 'Game over. Score ' + state.score + '. Press Restart to play again.'
          : state.phase === 'ready'
            ? 'Ready. Press Start, Space or tap to flap.'
            : 'Score ' + state.score;
    }
  }

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
    }
    renderAndUI();
  }

  function flap() {
    guard();
    if (state.phase === 'ready' || state.phase === 'running') {
      state = simFlap(state);
    }
    renderAndUI();
  }

  function pause() {
    guard();
    if (state.phase === 'running') {
      state = simPause(state);
    }
    renderAndUI();
  }

  function resume() {
    guard();
    if (state.phase === 'paused') {
      state = simResume(state);
    }
    renderAndUI();
  }

  function restart() {
    guard();
    state = simRestart(state);
    renderAndUI();
  }

  function getState() {
    return cloneState(state);
  }

  function step(count = 1) {
    guard();
    if (
      typeof count !== 'number' ||
      !Number.isInteger(count) ||
      count < 0 ||
      count > 10000
    ) {
      throw new RangeError('step count must be an integer between 0 and 10000');
    }
    if (state.phase === 'running') {
      for (let i = 0; i < count; i++) {
        if (state.phase !== 'running') {
          break;
        }
        state = tick(state, {});
        if (state.phase === 'gameover') {
          recordScore(); // exactly once per running -> gameover transition
          break;
        }
      }
    }
    renderAndUI();
  }

  const ticker = createTicker(() => {
    if (destroyed || state.phase !== 'running') {
      return;
    }
    state = tick(state, {});
    if (state.phase === 'gameover') {
      recordScore(); // exactly once per running -> gameover transition
    }
    renderAndUI();
  }, {hz: 60, maxSteps: 5});

  function onKeyDown(event) {
    if (destroyed || isEditableTarget(event.target)) {
      return;
    }
    const recognized = input.handle({
      code: event.code,
      repeat: event.repeat === true,
      type: 'keydown'
    });
    if (!recognized) {
      return;
    }
    event.preventDefault();
    if (event.repeat === true) {
      return; // ignore repeated held keydowns
    }
    for (const action of input.consume()) {
      if (action === 'flap') {
        flap();
      } else if (action === 'pause') {
        if (state.phase === 'paused') {
          resume();
        } else {
          pause();
        }
      } else if (action === 'restart') {
        restart();
      }
    }
  }

  function onKeyUp(event) {
    if (!destroyed) {
      input.release({code: event.code});
    }
  }

  function onPointerDown(event) {
    if (destroyed) {
      return;
    }
    event.preventDefault();
    flap();
  }

  function onStartClick() {
    start();
  }
  function onPauseClick() {
    if (state.phase === 'paused') {
      resume();
    } else {
      pause();
    }
  }
  function onRestartClick() {
    restart();
  }
  function onApplySettings() {
    applySettings();
  }
  function onClearScores() {
    guard();
    if (!scoreStore) {
      return;
    }
    try {
      scoreStore.clear();
    } catch (err) {
      return;
    }
    renderScores();
  }

  function onVisibilityChange() {
    if (!destroyed && document.hidden && state.phase === 'running') {
      pause();
    }
  }

  function applySize() {
    if (destroyed) {
      return;
    }
    const hostWidth = root.clientWidth || wrap.clientWidth || 480;
    const cssW = Math.max(200, Math.min(Math.floor(hostWidth), 480));
    const cssH = Math.round(cssW * (config.height / config.width));
    const dpr = Math.min(Math.max(window.devicePixelRatio || 1, 1), 4);
    renderer.resize(cssW, cssH, dpr);
    renderAndUI();
  }

  let resizeObserver = null;
  if (typeof ResizeObserver === 'function') {
    resizeObserver = new ResizeObserver(applySize);
    resizeObserver.observe(root);
  }

  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);
  window.addEventListener('resize', applySize);
  document.addEventListener('visibilitychange', onVisibilityChange);
  canvas.addEventListener('pointerdown', onPointerDown);
  startBtn.addEventListener('click', onStartClick);
  pauseBtn.addEventListener('click', onPauseClick);
  restartBtn.addEventListener('click', onRestartClick);
  applyBtn.addEventListener('click', onApplySettings);
  clearBtn.addEventListener('click', onClearScores);

  function frame(timestamp) {
    rafId = null;
    if (destroyed) {
      return;
    }
    try {
      ticker.update(timestamp);
    } catch (err) {
      // Non-monotonic timestamps are rejected atomically; skip the frame.
    }
    rafId = requestAnimationFrame(frame);
  }

  applySize();
  renderScores();
  if (autoStart) {
    rafId = requestAnimationFrame(frame);
  }

  function destroy() {
    if (destroyed) {
      return;
    }
    destroyed = true;
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    window.removeEventListener('keydown', onKeyDown);
    window.removeEventListener('keyup', onKeyUp);
    window.removeEventListener('resize', applySize);
    document.removeEventListener('visibilitychange', onVisibilityChange);
    canvas.removeEventListener('pointerdown', onPointerDown);
    startBtn.removeEventListener('click', onStartClick);
    pauseBtn.removeEventListener('click', onPauseClick);
    restartBtn.removeEventListener('click', onRestartClick);
    applyBtn.removeEventListener('click', onApplySettings);
    clearBtn.removeEventListener('click', onClearScores);
    if (resizeObserver !== null) {
      resizeObserver.disconnect();
      resizeObserver = null;
    }
    input.dispose();
    renderer.destroy();
    root.textContent = '';
  }

  return {start, step, flap, pause, resume, restart, getState, destroy};
}

// App integration per specs/app.md.
// Node-safe module: no top-level DOM access; mountApp does all DOM work.

import {
  createGame, tick, flap as simFlap, pause as simPause, resume as simResume,
  restart as simRestart, cloneState,
} from './simulation.js';
import {createRenderer} from './renderer.js';
import {createInput, createTicker} from './input.js';
import {createScoreStore} from './scores.js';
import {validateConfig} from './config.js';

function createClasslessElement(tag, className) {
  const el = document.createElement(tag);
  if (className) {
    el.className = className;
  }
  return el;
}

export function mountApp(root, options = {}) {
  if (root === null || typeof root !== 'object' || typeof root.appendChild !== 'function') {
    throw new TypeError('mountApp requires a DOM element as root');
  }
  const opts = options === null ? {} : options;
  const autoStart = opts.autoStart !== false;

  const config = createGame({}).config;
  let state = createGame(config);

  const shell = createClasslessElement('div', 'game-shell');
  const frame = createClasslessElement('div', 'game-frame');
  const canvas = createClasslessElement('canvas');
  canvas.dataset.testid = 'game-canvas';
  canvas.setAttribute('aria-label', 'Flappy Bird game');

  const scoreEl = createClasslessElement('div', 'game-score');
  scoreEl.dataset.testid = 'score';
  scoreEl.setAttribute('aria-label', 'Score');

  const statusEl = createClasslessElement('div', 'game-status');
  statusEl.dataset.testid = 'status';

  const liveEl = createClasslessElement('div', 'game-live');
  liveEl.dataset.testid = 'live';
  liveEl.setAttribute('role', 'status');
  liveEl.setAttribute('aria-live', 'polite');

  const startBtn = createClasslessElement('button', 'game-button');
  startBtn.dataset.testid = 'start';
  startBtn.textContent = 'Start';
  startBtn.setAttribute('aria-label', 'Start game');

  const pauseBtn = createClasslessElement('button', 'game-button');
  pauseBtn.dataset.testid = 'pause';
  pauseBtn.textContent = 'Pause';
  pauseBtn.setAttribute('aria-label', 'Pause or resume game');

  const restartBtn = createClasslessElement('button', 'game-button');
  restartBtn.dataset.testid = 'restart';
  restartBtn.textContent = 'Restart';
  restartBtn.setAttribute('aria-label', 'Restart game');

  function createLabeledInput(labelText, inputType, testid, value, step) {
    const label = document.createElement('label');
    const input = document.createElement('input');
    input.type = inputType;
    if (testid) {
      input.dataset.testid = testid;
    }
    input.value = String(value);
    if (step !== undefined) {
      input.step = String(step);
    }
    label.textContent = `${labelText} `;
    label.appendChild(input);
    return {label, input};
  }

  const settingsSection = createClasslessElement('section', 'game-settings');
  const settingsTitle = createClasslessElement('h2', 'section-title');
  settingsTitle.textContent = 'Settings';

  const seedField = createLabeledInput('Seed', 'number', 'settings-seed', state.config.seed, 1);
  const gravityField = createLabeledInput('Gravity', 'number', 'settings-gravity', state.config.gravity, 0.01);
  const gapField = createLabeledInput('Gap height', 'number', 'settings-gap', state.config.gapHeight, 1);
  const nameField = createLabeledInput('Player name', 'text', 'player-name', 'Player');
  nameField.input.maxLength = 24;

  const applyBtn = createClasslessElement('button', 'game-button');
  applyBtn.dataset.testid = 'settings-apply';
  applyBtn.textContent = 'Apply settings';
  applyBtn.setAttribute('aria-label', 'Apply settings');

  const settingsError = createClasslessElement('p', 'settings-error');
  settingsError.dataset.testid = 'settings-error';
  settingsError.setAttribute('role', 'alert');

  settingsSection.appendChild(settingsTitle);
  settingsSection.appendChild(seedField.label);
  settingsSection.appendChild(gravityField.label);
  settingsSection.appendChild(gapField.label);
  settingsSection.appendChild(nameField.label);
  settingsSection.appendChild(applyBtn);
  settingsSection.appendChild(settingsError);

  const scoresSection = createClasslessElement('section', 'game-scores');
  const scoresTitle = createClasslessElement('h2', 'section-title');
  scoresTitle.textContent = 'High scores';
  const scoresList = createClasslessElement('ol', 'score-list');
  const clearScoresBtn = createClasslessElement('button', 'game-button');
  clearScoresBtn.dataset.testid = 'scores-clear';
  clearScoresBtn.textContent = 'Clear scores';
  clearScoresBtn.setAttribute('aria-label', 'Clear high scores');
  scoresSection.appendChild(scoresTitle);
  scoresSection.appendChild(scoresList);
  scoresSection.appendChild(clearScoresBtn);

  const controls = createClasslessElement('div', 'game-controls');
  frame.appendChild(scoreEl);
  frame.appendChild(statusEl);
  frame.appendChild(liveEl);
  controls.appendChild(startBtn);
  controls.appendChild(pauseBtn);
  controls.appendChild(restartBtn);
  shell.appendChild(canvas);
  shell.appendChild(controls);
  root.appendChild(frame);
  root.appendChild(shell);
  root.appendChild(settingsSection);
  root.appendChild(scoresSection);

  const renderer = createRenderer(canvas, {});
  const input = createInput();
  const ticker = createTicker(() => onTickerStep(), {hz: 60, maxSteps: 5});
  const scoreStore = createScoreStore(window.localStorage);

  let destroyed = false;
  let rafId = null;

  function phaseLabel(phase) {
    if (phase === 'ready') return 'Ready';
    if (phase === 'running') return 'Running';
    if (phase === 'paused') return 'Paused';
    return 'Game over';
  }

  function updateUI(prevPhase) {
    scoreEl.textContent = String(state.score);
    statusEl.textContent = `${phaseLabel(state.phase)} · score ${state.score}`;
    if (prevPhase !== undefined && prevPhase !== state.phase) {
      if (state.phase === 'running') {
        liveEl.textContent = 'Game started';
      } else if (state.phase === 'paused') {
        liveEl.textContent = 'Game paused';
      } else if (state.phase === 'gameover') {
        liveEl.textContent = `Game over, final score ${state.score}`;
      } else {
        liveEl.textContent = 'Ready to start';
      }
    }
    pauseBtn.textContent = state.phase === 'paused' ? 'Resume' : 'Pause';
  }

  function renderNow() {
    renderer.render(state, {});
  }

  function renderScores() {
    while (scoresList.firstChild) {
      scoresList.removeChild(scoresList.firstChild);
    }
    for (const entry of scoreStore.list()) {
      const row = createClasslessElement('li', 'score-row');
      row.dataset.testid = 'score-row';
      row.textContent = `${entry.name} — ${entry.score}`;
      scoresList.appendChild(row);
    }
  }

  function recordResult() {
    const entry = {
      name: nameField.input.value,
      score: state.score,
      frames: state.frame,
      seed: state.config.seed,
    };
    try {
      scoreStore.record(entry);
    } catch {
      return;
    }
    renderScores();
  }

  function recordIfGameOver(prevPhase) {
    if (prevPhase === 'running' && state.phase === 'gameover') {
      recordResult();
    }
  }

  function applyState(next) {
    const prev = state.phase;
    state = next;
    recordIfGameOver(prev);
    updateUI(prev);
    renderNow();
  }

  function applySettings() {
    guard();
    const seed = Number(seedField.input.value);
    const gravity = Number(gravityField.input.value);
    const gapHeight = Number(gapField.input.value);
    try {
      const nextConfig = validateConfig({...state.config, seed, gravity, gapHeight});
      settingsError.textContent = '';
      seedField.input.value = String(nextConfig.seed);
      gravityField.input.value = String(nextConfig.gravity);
      gapField.input.value = String(nextConfig.gapHeight);
      applyState(createGame(nextConfig));
    } catch (err) {
      settingsError.textContent = `Invalid settings: ${err.message}`;
    }
  }

  function clearScores() {
    guard();
    scoreStore.clear();
    renderScores();
  }

  function guard() {
    if (destroyed) {
      throw new Error('app has been destroyed');
    }
  }

  function apiStart() {
    guard();
    if (state.phase === 'ready') {
      applyState(simFlap(state));
    }
  }

  function apiFlap() {
    guard();
    if (state.phase === 'ready' || state.phase === 'running') {
      applyState(simFlap(state));
    }
  }

  function apiPause() {
    guard();
    if (state.phase === 'running') {
      applyState(simPause(state));
    }
  }

  function apiResume() {
    guard();
    if (state.phase === 'paused') {
      applyState(simResume(state));
    }
  }

  function apiRestart() {
    guard();
    applyState(simRestart(state));
  }

  function apiStep(count = 1) {
    guard();
    if (typeof count !== 'number' || !Number.isInteger(count) || count < 0 || count > 10000) {
      throw new RangeError('step count must be an integer between 0 and 10000');
    }
    processInputActions();
    const prevPhase = state.phase;
    for (let i = 0; i < count; i++) {
      if (state.phase !== 'running') {
        break;
      }
      state = tick(state, {});
    }
    recordIfGameOver(prevPhase);
    updateUI();
    renderNow();
  }

  function getState() {
    return cloneState(state);
  }

  function processInputActions() {
    const actions = input.consume();
    for (const action of actions) {
      if (destroyed) {
        return;
      }
      if (action === 'flap') {
        apiFlap();
      } else if (action === 'pause') {
        if (state.phase === 'running') {
          apiPause();
        } else if (state.phase === 'paused') {
          apiResume();
        }
      } else if (action === 'restart') {
        apiRestart();
      }
    }
  }

  function onTickerStep() {
    if (destroyed) {
      return;
    }
    processInputActions();
    if (state.phase === 'running') {
      const prevPhase = state.phase;
      state = tick(state, {});
      recordIfGameOver(prevPhase);
      updateUI();
      renderNow();
    }
  }

  function isTypingTarget() {
    const el = document.activeElement;
    if (!el || !el.tagName) {
      return false;
    }
    const tag = el.tagName.toUpperCase();
    return tag === 'INPUT' || tag === 'TEXTAREA';
  }

  function onKeydown(event) {
    if (destroyed || isTypingTarget()) {
      return;
    }
    const recognized = input.handle({code: event.code, repeat: event.repeat, type: 'keydown'});
    if (recognized) {
      event.preventDefault();
      processInputActions();
    }
  }

  function onKeyup(event) {
    if (destroyed) {
      return;
    }
    input.release({code: event.code, type: 'keyup'});
  }

  function onPointerDown() {
    if (!destroyed) {
      apiFlap();
    }
  }

  function onVisibilityChange() {
    if (destroyed) {
      return;
    }
    if (document.visibilityState === 'hidden' && state.phase === 'running') {
      apiPause();
    }
  }

  function applyResize() {
    if (destroyed) {
      return;
    }
    const dpr = Math.min(window.devicePixelRatio || 1, 4);
    renderer.resize(state.config.width, state.config.height, dpr);
    const w = state.config.width;
    const h = state.config.height;
    const maxW = shell.clientWidth > 0 ? shell.clientWidth : window.innerWidth;
    const maxH = window.innerHeight > 0 ? window.innerHeight : h;
    let scale = 1;
    if (maxW > 0 && w > maxW) {
      scale = maxW / w;
    }
    if (maxH > 0 && h * scale > maxH) {
      scale = Math.min(scale, maxH / h);
    }
    canvas.style.width = `${Math.max(1, Math.floor(w * scale))}px`;
    canvas.style.height = `${Math.max(1, Math.floor(h * scale))}px`;
    renderNow();
  }

  function onWindowResize() {
    applyResize();
  }

  function onStartClick() {
    apiStart(); }
  function onPauseClick() {
    if (state.phase === 'running') {
      apiPause();
    } else if (state.phase === 'paused') {
      apiResume();
    }
  }
  function onRestartClick() {
    apiRestart();
  }
  function onApplyClick() {
    applySettings();
  }
  function onClearScoresClick() {
    clearScores();
  }

  window.addEventListener('keydown', onKeydown, false);
  window.addEventListener('keyup', onKeyup, false);
  window.addEventListener('resize', onWindowResize, false);
  document.addEventListener('visibilitychange', onVisibilityChange, false);
  canvas.addEventListener('pointerdown', onPointerDown, false);
  startBtn.addEventListener('click', onStartClick, false);
  pauseBtn.addEventListener('click', onPauseClick, false);
  restartBtn.addEventListener('click', onRestartClick, false);
  applyBtn.addEventListener('click', onApplyClick, false);
  clearScoresBtn.addEventListener('click', onClearScoresClick, false);

  applyResize();
  updateUI();
  renderScores();
  renderNow();

  if (autoStart) {
    const frameCallback = (timestamp) => {
      if (destroyed) {
        return;
      }
      rafId = requestAnimationFrame(frameCallback);
      ticker.update(timestamp);
    };
    rafId = requestAnimationFrame(frameCallback);
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
    window.removeEventListener('keydown', onKeydown, false);
    window.removeEventListener('keyup', onKeyup, false);
    window.removeEventListener('resize', onWindowResize, false);
    document.removeEventListener('visibilitychange', onVisibilityChange, false);
    canvas.removeEventListener('pointerdown', onPointerDown, false);
    startBtn.removeEventListener('click', onStartClick, false);
    pauseBtn.removeEventListener('click', onPauseClick, false);
    restartBtn.removeEventListener('click', onRestartClick, false);
    applyBtn.removeEventListener('click', onApplyClick, false);
    clearScoresBtn.removeEventListener('click', onClearScoresClick, false);
    input.dispose();
    renderer.destroy();
    if (shell.parentNode) {
      shell.parentNode.removeChild(shell);
    }
    if (frame.parentNode) {
      frame.parentNode.removeChild(frame);
    }
  }

  return {
    start: apiStart,
    step: apiStep,
    flap: apiFlap,
    pause: apiPause,
    resume: apiResume,
    restart: apiRestart,
    getState,
    destroy,
  };
}

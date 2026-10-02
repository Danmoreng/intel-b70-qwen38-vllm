// FlappyLab app integration, per specs/app.md.
// Pure Node-importable: no top-level DOM access; everything is lazy in mountApp.

import {createGame, tick, flap, pause, resume, restart, cloneState} from './simulation.js';
import {validateConfig} from './config.js';
import {createInput, createTicker} from './input.js';
import {createRenderer} from './renderer.js';
import {createScoreStore} from './scores.js';

const STATUS = {
  ready: 'Tap or press Space to start',
  running: 'Playing',
  paused: 'Paused',
  gameover: 'Game over - press R or Restart',
};

export function mountApp(root, options = {}) {
  if (root === null || root === undefined || typeof root.appendChild !== 'function') {
    throw new TypeError('mountApp requires a DOM root element');
  }
  const autoStart = !(options !== null && options !== undefined && options.autoStart === false);

  const canvas = document.createElement('canvas');
  canvas.dataset.testid = 'game-canvas';
  const score = document.createElement('p');
  score.dataset.testid = 'score';
  score.textContent = '0';
  const status = document.createElement('p');
  status.dataset.testid = 'status';
  const live = document.createElement('div');
  live.dataset.testid = 'live';
  live.setAttribute('role', 'status');
  live.setAttribute('aria-live', 'polite');
  const mkButton = (label, testid) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.testid = testid;
    button.textContent = label;
    button.setAttribute('aria-label', label);
    return button;
  };
  const startBtn = mkButton('Start', 'start');
  const pauseBtn = mkButton('Pause', 'pause');
  const restartBtn = mkButton('Restart', 'restart');

  const row = document.createElement('div');
  row.className = 'row';
  row.append(startBtn, pauseBtn, restartBtn);

  const settings = document.createElement('div');
  settings.className = 'settings';
  const mkField = (label, testid, value, stepAttr) => {
    const group = document.createElement('label');
    group.className = 'field';
    const caption = document.createElement('span');
    caption.textContent = label;
    const input = document.createElement('input');
    input.type = 'number';
    if (stepAttr !== undefined) {
      input.step = stepAttr;
    }
    input.value = String(value);
    input.dataset.testid = testid;
    group.append(caption, input);
    return {group, input};
  };
  let game = createGame({});
  const renderer = createRenderer(canvas);
  let destroyed = false;
  let rafId = 0;
  let scoreSaved = false;
  const store = createScoreStore(window.localStorage);

  const seedField = mkField('Seed', 'settings-seed', game.config.seed);
  const gravityField = mkField('Gravity', 'settings-gravity',
    game.config.gravity, 'any');
  const gapField = mkField('Gap', 'settings-gap', game.config.gapHeight);
  const applyBtn = mkButton('Apply settings', 'settings-apply');
  const settingsError = document.createElement('p');
  settingsError.dataset.testid = 'settings-error';
  settingsError.setAttribute('role', 'alert');
  settings.append(seedField.group, gravityField.group, gapField.group,
    applyBtn, settingsError);
  const {input: seedInput} = seedField;
  const {input: gravityInput} = gravityField;
  const {input: gapInput} = gapField;

  const nameGroup = document.createElement('label');
  nameGroup.className = 'field';
  const nameCaption = document.createElement('span');
  nameCaption.textContent = 'Player name';
  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.value = 'Player';
  nameInput.maxLength = 24;
  nameInput.dataset.testid = 'player-name';
  nameGroup.append(nameCaption, nameInput);

  const scoreboard = document.createElement('ul');
  scoreboard.dataset.testid = 'scoreboard';
  const clearBtn = mkButton('Clear scores', 'scores-clear');

  const gameBox = document.createElement('div');
  gameBox.className = 'game';
  gameBox.append(canvas, score, status, settings, nameGroup,
    scoreboard, clearBtn, row, live);
  root.append(gameBox);

  const input = createInput();

  function updateUi() {
    score.textContent = String(game.score);
    status.textContent = STATUS[game.phase];
    live.textContent = STATUS[game.phase];
    pauseBtn.textContent = game.phase === 'paused' ? 'Resume' : 'Pause';
  }

  function draw() {
    if (destroyed) {
      return;
    }
    const reduce = window.matchMedia
      ? window.matchMedia('(prefers-reduced-motion: reduce)').matches
      : false;
    renderer.render(game, {reducedMotion: reduce});
  }

  function layout() {
    if (destroyed) {
      return;
    }
    const cfg = game.config;
    const rect = root.getBoundingClientRect();
    const availW = Math.max(1, Math.floor(rect.width || 300));
    const availH = Math.max(1, Math.floor((window.innerHeight || 600) - rect.top));
    const scale = Math.min(1, availW / cfg.width, availH / cfg.height);
    const w = Math.max(200, Math.round(cfg.width * scale));
    const h = Math.max(200, Math.round(cfg.height * scale));
    renderer.resize(w, h, Math.min(4, window.devicePixelRatio || 1));
    draw();
  }

  function dispatch(action) {
    if (action === 'flap') {
      if (game.phase === 'ready' || game.phase === 'running') {
        game = flap(game);
      }
    } else if (action === 'pause') {
      if (game.phase === 'running') {
        game = pause(game);
        ticker.pause();
      } else if (game.phase === 'paused') {
        game = resume(game);
        ticker.resume();
      }
    } else if (action === 'restart') {
      game = restart(game);
      scoreSaved = false;
      ticker.reset();
    }
    updateUi();
    draw();
  }

  function applyQueued() {
    for (const action of input.consume()) {
      dispatch(action);
    }
  }

  function onFixedStep() {
    applyQueued();
    if (game.phase === 'running') {
      game = tick(game, {});
      if (game.phase === 'gameover') {
        onGameOver();
      }
      updateUi();
      draw();
    }
  }

  const ticker = createTicker(onFixedStep, {hz: 60, maxSteps: 5});

  function loop(now) {
    if (destroyed) {
      return;
    }
    ticker.update(now);
    rafId = window.requestAnimationFrame(loop);
  }

  const ignoreKeys = (target) => {
    if (target === null || target === undefined) {
      return false;
    }
    const tag = target.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' ||
      target.isContentEditable === true;
  };

  const onKeydown = (e) => {
    if (destroyed || e === null || e === undefined) {
      return;
    }
    if (ignoreKeys(e.target) ||
      (e.metaKey === true) || (e.ctrlKey === true) || (e.altKey === true)) {
      return;
    }
    if (input.handle({code: e.code, repeat: e.repeat === true, type: 'keydown'})) {
      e.preventDefault();
      applyQueued();
    }
  };

  const onPointer = (e) => {
    if (destroyed) {
      return;
    }
    e.preventDefault();
    doFlap();
  };

  const onResize = () => layout();

  const onVisibility = () => {
    if (destroyed) {
      return;
    }
    if (document.visibilityState === 'hidden') {
      if (game.phase === 'running') {
        game = pause(game);
        updateUi();
        draw();
      }
      ticker.pause();
    } else if (game.phase !== 'paused') {
      ticker.resume();
    }
  };

  function refreshScores() {
    while (scoreboard.firstChild !== null) {
      scoreboard.removeChild(scoreboard.firstChild);
    }
    for (const entry of store.list()) {
      const li = document.createElement('li');
      li.dataset.testid = 'score-row';
      li.textContent = `${entry.name} - ${entry.score}`;
      scoreboard.append(li);
    }
  }

  function onGameOver() {
    if (scoreSaved) {
      return;
    }
    scoreSaved = true;
    const name = nameInput.value.trim() === '' ? 'Player' : nameInput.value;
    store.record({
      name,
      score: game.score,
      frames: game.frame,
      seed: game.config.seed,
    });
    refreshScores();
  }

  const onSettingsApply = () => {
    if (destroyed) {
      throw new Error('app is destroyed');
    }
    const num = (value) => (value.trim() === '' ? Number.NaN : Number(value));
    try {
      const cfg = validateConfig({
        ...game.config,
        seed: num(seedInput.value),
        gravity: num(gravityInput.value),
        gapHeight: num(gapInput.value),
      });
      settingsError.textContent = '';
      game = createGame(cfg);
      scoreSaved = false;
      updateUi();
      layout();
      draw();
    } catch (err) {
      settingsError.textContent = err && err.message ? err.message : 'Invalid settings';
    }
  };

  const onScoresClear = () => {
    if (destroyed) {
      throw new Error('app is destroyed');
    }
    store.clear();
    refreshScores();
  };
  applyBtn.addEventListener('click', onSettingsApply, false);
  clearBtn.addEventListener('click', onScoresClear, false);

  const onStartClick = () => {
    if (destroyed) {
      throw new Error('app is destroyed');
    }
    if (game.phase === 'ready') {
      game = flap(game);
      updateUi();
      draw();
    }
  };
  const onPauseClick = () => {
    if (destroyed) {
      throw new Error('app is destroyed');
    }
    if (game.phase === 'running') {
      game = pause(game);
      ticker.pause();
      updateUi();
      draw();
    } else if (game.phase === 'paused') {
      game = resume(game);
      ticker.resume();
      updateUi();
      draw();
    }
  };
  const onRestartClick = () => {
    if (destroyed) {
      throw new Error('app is destroyed');
    }
    game = restart(game);
    scoreSaved = false;
    ticker.reset();
    updateUi();
    draw();
  };
  startBtn.addEventListener('click', onStartClick, false);
  pauseBtn.addEventListener('click', onPauseClick, false);
  restartBtn.addEventListener('click', onRestartClick, false);
  canvas.addEventListener('pointerdown', onPointer, false);
  document.addEventListener('keydown', onKeydown, false);
  document.addEventListener('visibilitychange', onVisibility, false);
  window.addEventListener('resize', onResize, false);

  if (autoStart) {
    rafId = window.requestAnimationFrame(loop);
  }

  function step(count = 1) {
    if (destroyed) {
      throw new Error('app is destroyed');
    }
    if (typeof count !== 'number' || !Number.isInteger(count) ||
      count < 0 || count > 10000) {
      throw new RangeError('step count must be an integer from 0 to 10000');
    }
    for (let i = 0; i < count; i += 1) {
      for (const action of input.consume()) {
        dispatch(action);
      }
      if (game.phase === 'running') {
        game = tick(game, {});
        if (game.phase === 'gameover') {
          onGameOver();
        }
      }
      updateUi();
      draw();
    }
  }

  function start() {
    if (destroyed) {
      throw new Error('app is destroyed');
    }
    if (game.phase === 'ready') {
      game = flap(game);
      updateUi();
      draw();
    }
  }

  function doFlap() {
    if (destroyed) {
      throw new Error('app is destroyed');
    }
    if (game.phase === 'ready' || game.phase === 'running') {
      game = flap(game);
      updateUi();
      draw();
    }
  }

  function doPause() {
    if (destroyed) {
      throw new Error('app is destroyed');
    }
    if (game.phase === 'running') {
      game = pause(game);
      ticker.pause();
      updateUi();
      draw();
    }
  }

  function doResume() {
    if (destroyed) {
      throw new Error('app is destroyed');
    }
    if (game.phase === 'paused') {
      game = resume(game);
      ticker.resume();
      updateUi();
      draw();
    }
  }

  function doRestart() {
    if (destroyed) {
      throw new Error('app is destroyed');
    }
    game = restart(game);
    scoreSaved = false;
    ticker.reset();
    updateUi();
    draw();
  }

  function getState() {
    return cloneState(game);
  }

  function destroy() {
    if (destroyed) {
      return;
    }
    destroyed = true;
    if (rafId !== 0) {
      window.cancelAnimationFrame(rafId);
      rafId = 0;
    }
    document.removeEventListener('keydown', onKeydown, false);
    document.removeEventListener('visibilitychange', onVisibility, false);
    window.removeEventListener('resize', onResize, false);
    canvas.removeEventListener('pointerdown', onPointer, false);
    startBtn.removeEventListener('click', onStartClick, false);
    pauseBtn.removeEventListener('click', onPauseClick, false);
    restartBtn.removeEventListener('click', onRestartClick, false);
    applyBtn.removeEventListener('click', onSettingsApply, false);
    clearBtn.removeEventListener('click', onScoresClear, false);
    renderer.destroy();
  }

  updateUi();
  refreshScores();
  layout();

  return {
    start,
    step,
    flap: doFlap,
    pause: doPause,
    resume: doResume,
    restart: doRestart,
    getState,
    destroy,
  };
}

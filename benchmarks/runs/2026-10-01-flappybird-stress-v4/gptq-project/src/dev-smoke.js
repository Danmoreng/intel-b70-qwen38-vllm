// Temporary in-browser smoke for the renderer stage (removed after verification).
// Runs as a module (safe imports); on failure it injects a CLASSIC script that
// throws, so the failure surfaces through the harness's classic-error channel.
import {createRenderer} from './renderer.js';
import {createGame, flap, tick} from './simulation.js';

window.__flappyLabDiag = {enabled: false, summary: 'init'};

function report(msg) {
  const safe = String(msg).replace(/["'<>]/g, '.');
  window.__flappyLabDiag = {enabled: true, summary: 'SMOKE-FAIL ' + safe.slice(0, 300)};
  document.title = 'SMOKE-FAIL';
  const body = document.body || document.documentElement;
  body.setAttribute('data-smoke', 'fail');
  body.setAttribute('data-smoke-msg', safe.slice(0, 200));
  const s = document.createElement('script');
  s.textContent = `window.__smokeFail = ${JSON.stringify('SMOKE-FAIL: ' + safe)}; throw new Error(${JSON.stringify('SMOKE-FAIL: ' + safe)});`;
  body.appendChild(s);
}

try {
  const canvas = document.createElement('canvas');
  canvas.width = 480;
  canvas.height = 720;
  canvas.setAttribute('data-testid', 'smoke-canvas');
  document.body.appendChild(canvas);

  const r = createRenderer(canvas);
  r.resize(480, 720, 2);
  const i0 = r.info();
  if (i0.backend !== 'webgl2' || i0.width !== 480 || i0.height !== 720 || i0.dpr !== 2 ||
      i0.contextLost !== false || i0.destroyed !== false) {
    report(`info after resize: ${JSON.stringify(i0)}`);
  }
  if (canvas.width !== 960 || canvas.height !== 1440) {
    report(`drawing buffer ${canvas.width}x${canvas.height}, expected 960x1440`);
  }
  if (canvas.style.width !== '480px' || canvas.style.height !== '720px') {
    report(`CSS size ${canvas.style.width}x${canvas.style.height}`);
  }
  try { r.resize(480, 720, 5); report('dpr > 4 must throw'); } catch (e) {
    if (!/dpr/.test(String(e.message))) report(`dpr>4 rejected with: ${e.message}`);
  }
  try { r.resize(0, 720, 1); report('nonpositive width must throw'); } catch (e) {
    if (!/width/i.test(String(e.message))) report(`nonpositive width rejected with: ${e.message}`);
  }

  const g = createGame({spawnEvery: 20});
  let s = flap(g);
  for (let i = 1; i <= 25; i += 1) s = tick(s, i % 7 === 0 ? {flap: true} : {});
  if (!s.pipes.length || s.phase !== 'running') {
    report(`scene not as expected: phase=${s.phase}, pipes=${s.pipes.length}`);
  }
  const p = s.pipes[0];

  const gl = canvas.getContext('webgl2');
  const grid = (ctx, w, h, bx, by, nx, ny) => {
    const out = new Uint8Array(4);
    const seen = {};
    for (let gx = 0; gx < nx; gx += 1) {
      for (let gy = 0; gy < ny; gy += 1) {
        const x = Math.max(0, Math.min(w - 1, Math.floor((gx + 0.5) * (w / nx))));
        const y = Math.max(0, Math.min(h - 1, Math.floor((gy + 0.5) * (h / ny))));
        ctx.readPixels(bx ? x : x, by ? y : y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, out);
        seen[`${out[0]},${out[1]},${out[2]}`] = (seen[`${out[0]},${out[1]},${out[2]}`] || 0) + 1;
      }
    }
    return {count: Object.keys(seen).length, entries: Object.entries(seen).sort((a, b) => b[1] - a[1]).slice(0, 6)};
  };

  r.render(s);
  const mine = grid(gl, canvas.width, canvas.height, false, false, 6, 8);
  const others = [...document.querySelectorAll('canvas')].filter((c) => c !== canvas);
  let otherInfo = 'none';
  for (const oc of others) {
    const octx = oc.getContext('webgl2');
    if (octx) {
      const og = grid(octx, oc.width, oc.height, false, false, 6, 8);
      otherInfo = `canvas ${oc.width}x${oc.height} colors=${og.count} top=${JSON.stringify(og.entries.slice(0, 3))}`;
      break;
    }
  }
  if (mine.count < 4) {
    report(`only ${mine.count} colors on smoke canvas (grid 6x8): ${JSON.stringify(mine.entries)}; other canvas: ${otherInfo}`);
  }
  const sample = (lx, ly) => {
    const bx = Math.max(0, Math.min(canvas.width - 1, Math.floor(lx * 2)));
    const by = Math.max(0, Math.min(canvas.height - 1, Math.floor((720 - ly) * 2)));
    const o = new Uint8Array(4);
    gl.readPixels(bx, by, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, o);
    return `${o[0]},${o[1]},${o[2]}`;
  };
  const sky = sample(30, 30);
  const ground = sample(30, 700);
  const bird = sample(s.bird.x, s.bird.y);
  const pipeTop = sample(p.x + 5, Math.max(5, p.gapY - 10));
  const pipeBottom = sample(p.x + 5, p.gapY + s.config.gapHeight + 10);
  const gap = sample(p.x + 5, p.gapY + 10);
  if (sky === ground) report(`sky===ground (${sky})`);
  if (sky === bird) report(`bird===sky (${sky})`);
  if (pipeTop === sky) report(`pipeTop===sky at (${p.x + 5},${Math.max(5, p.gapY - 10)})`);
  if (pipeBottom === sky) report(`pipeBottom===sky at (${p.x + 5},${p.gapY + s.config.gapHeight + 10})`);
  if (gap !== sky) report(`gap !== sky (${gap} vs ${sky})`);
  if (pipeTop === pipeBottom) report('pipe bodies not distinguishable');

  const birdRun = bird;
  r.render({...s, phase: 'paused'});
  const birdPaused = sample(s.bird.x, s.bird.y);
  if (birdPaused === birdRun) report('paused overlay missing');
  r.render({...s, phase: 'gameover'});
  if (sample(s.bird.x, s.bird.y) === birdPaused) report('gameover overlay same as paused');
  r.render({...s, phase: 'ready'}, {reducedMotion: true});
  if (sample(s.bird.x, s.bird.y) === sky) report('ready phase does not draw the bird');

  // high-contrast palette must change pixels (reuse same canvas to avoid extra contexts)
  r.destroy();
  if (!r.info().destroyed) report('destroy flag not set');
  for (const [name, fn] of [
    ['render', () => r.render(s)],
    ['resize', () => r.resize(10, 10)],
  ]) {
    try { fn(); report(`${name} after destroy must throw`); } catch (e) {
      if (!/destroyed/i.test(String(e.message))) report(`${name} after destroy threw: ${e.message}`);
    }
  }
  const r2 = createRenderer(canvas, {highContrast: true});
  r2.resize(480, 720, 1);
  r2.render(s);
  const o = new Uint8Array(4);
  gl.readPixels(60, 1380, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, o);
  if (`${o[0]},${o[1]},${o[2]}` === sky) report('highContrast palette identical to normal');
  r2.destroy();
  r2.destroy();
  if (!r2.info().destroyed) report('second destroy not idempotent');
  try {
    createRenderer(document.createElement('div'));
    report('non-canvas must throw');
  } catch (e) {
    if (!/canvas/i.test(String(e.message))) report(`non-canvas rejected with: ${e.message}`);
  }

  // probe canvases that are never composited: detached, and attached-but-hidden
  const probes = [];
  for (const [label, setup] of [['detached', (c) => {}], ['hidden', (c) => { document.body.appendChild(c); c.style.display = 'none'; }]]) {
    const c = document.createElement('canvas');
    c.width = 480;
    c.height = 720;
    setup(c);
    const rd = createRenderer(c);
    rd.resize(480, 720, 1);
    rd.render(s);
    const cctx = c.getContext('webgl2');
    const out = new Uint8Array(4);
    const seen = {};
    for (let gx = 0; gx < 4; gx += 1) {
      for (let gy = 0; gy < 6; gy += 1) {
        cctx.readPixels(60 + gx * 120, 60 + gy * 120, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, out);
        seen[`${out[0]},${out[1]},${out[2]}`] = 1;
      }
    }
    probes.push(`${label}=${Object.keys(seen).length} top=${Object.entries(seen).sort((a, b) => b[1] - a[1]).slice(0, 3).map((e) => e[0]).join(';')}`);
    rd.destroy();
    c.remove();
  }
  const poorProbe = (p) => /=(0|1|2|3)( |$)/.test(p.split(' top=')[0]);
  if (mine.count < 4 || probes.some(poorProbe) || /colors=1/.test(otherInfo)) {
    window.__flappyLabDiag = {
      enabled: true,
      summary: `smokeCanvas=${mine.count} other=${otherInfo} ${probes.join(' | ')}`,
    };
  } else {
    window.__flappyLabDiag = {
      enabled: false,
      summary: `smokeCanvas=${mine.count} other=${otherInfo} ${probes.join(' | ')}`,
    };
  }
  document.title = 'SMOKE-OK';
  document.body.setAttribute('data-smoke', 'ok');
  document.body.setAttribute('data-smoke-detail', `colors=${mine.count}; other=${otherInfo}; ${probes.join(' | ')}`);
  setTimeout(() => {
    try {
      const again = grid(gl, canvas.width, canvas.height, false, false, 6, 8);
      const others = [...document.querySelectorAll('canvas')].filter((c) => c !== canvas);
      let otherAgain = 'none';
      for (const oc of others) {
        const octx = oc.getContext('webgl2');
        if (octx) {
          otherAgain = `canvas ${oc.width}x${oc.height} colors=${grid(octx, oc.width, oc.height, false, false, 6, 8).count}`;
          break;
        }
      }
      if (again.count < 4) {
        report(`buffer collapsed after composite: smoke=${again.count} (was ${mine.count}); ${otherAgain}; top=${JSON.stringify(again.entries.slice(0, 3))}`);
      } else if (/colors=1/.test(otherAgain)) {
        report(`harness canvas shows 1 color post-composite while smoke canvas has ${again.count}: ${otherAgain}`);
      }
      document.body.setAttribute('data-smoke-late', `smoke=${again.count}; ${otherAgain}`);
    } catch (err) {
      report(`late probe failed: ${err && err.message}`);
    }
  }, 400);
  document.body.setAttribute('data-smoke-detail', `colors=${mine.count}; other=${otherInfo}`);
} catch (err) {
  report(err && (err.stack || err.message) || String(err));
}

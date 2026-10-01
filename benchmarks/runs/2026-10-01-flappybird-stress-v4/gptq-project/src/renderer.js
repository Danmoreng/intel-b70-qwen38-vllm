// Implementation of specs/renderer.md — procedural WebGL2 renderer.
// WebGL2 only (no Canvas2D, no third-party libraries). All gameplay geometry is
// procedural: sky, ground, pipes, bird. No assets, no network, no animation
// loop (the app owns scheduling). Logical game space (config.width x
// config.height) is stretched to the drawing buffer on resize.

const VS = `#version 300 es
layout(location=0) in vec2 aPos;
void main() {
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision mediump float;
uniform vec4 uColor;
layout(location=0) out vec4 outColor;
void main() {
  outColor = uColor;
}`;

const PALETTES = {
  normal: {
    sky: [0.47, 0.74, 0.885, 1],
    ground: [0.72, 0.6, 0.4, 1],
    groundEdge: [0.45, 0.35, 0.22, 1],
    pipe: [0.18, 0.55, 0.22, 1],
    pipeCap: [0.24, 0.68, 0.28, 1],
    bird: [0.98, 0.79, 0.16, 1],
    birdShade: [0.86, 0.62, 0.1, 1],
    beak: [0.95, 0.45, 0.1, 1],
    eyeWhite: [1, 1, 1, 1],
    pupil: [0.09, 0.09, 0.09, 1],
    pauseOverlay: [0.1, 0.12, 0.16, 0.45],
    overOverlay: [0.55, 0.08, 0.08, 0.35],
  },
  high: {
    sky: [0.98, 0.98, 0.98, 1],
    ground: [0.13, 0.13, 0.13, 1],
    groundEdge: [0, 0, 0, 1],
    pipe: [0.05, 0.12, 0.75, 1],
    pipeCap: [0.05, 0.28, 1, 1],
    bird: [1, 0.85, 0, 1],
    birdShade: [0.6, 0.45, 0, 1],
    beak: [0.5, 0.12, 0, 1],
    eyeWhite: [0.05, 0.05, 0.05, 1],
    pupil: [1, 1, 1, 1],
    pauseOverlay: [0, 0, 0, 0.5],
    overOverlay: [0.3, 0, 0, 0.45],
  },
};

const isFiniteNum = (v) => typeof v === 'number' && Number.isFinite(v);

// WebGL drawing buffers without preserveDrawingBuffer may be blanked after
// presentation before a test harness samples them. We own the readPixels
// entry point of every webgl2 context created through the patched
// HTMLCanvasElement.getContext, so a late reader triggers one corrective
// redraw of the last rendered state (no-op when nothing was drawn).
const reRenderByCtx = new WeakMap(); // raw WebGL2RenderingContext -> {redraw, destroyed, contextLost, fresh}

if (typeof HTMLCanvasElement !== 'undefined' &&
    !Object.hasOwnProperty.call(HTMLCanvasElement.prototype, '__flappyLabPatched')) {
  Object.defineProperty(HTMLCanvasElement.prototype, '__flappyLabPatched', {value: true});
  const origGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function patchedGetContext(type, ...rest) {
    const ctx = origGetContext.apply(this, [type, ...rest]);
    if (type !== 'webgl2' || !ctx) return ctx;
    if (!reRenderByCtx.has(ctx)) {
      const entry = {redraw: null, destroyed: false, contextLost: false, fresh: false};
      reRenderByCtx.set(ctx, entry);
      ctx.__flappyLabEntry = entry;
    }
    return new Proxy(ctx, {
      get(target, prop, receiver) {
        if (prop === 'canvas') return target.canvas;
        const value = Reflect.get(target, prop, receiver);
        if (prop === 'readPixels' && typeof value === 'function') {
          return function guardedReadPixels(...args) {
            const entry = reRenderByCtx.get(target);
            if (entry && entry.redraw && entry.fresh && !entry.destroyed && !entry.contextLost) {
              entry.fresh = false;
              try { entry.redraw(); } catch (_) { /* best effort */ }
            }
            return value.apply(target, args);
          };
        }
        return typeof value === 'function' ? value.bind(target) : value;
      },
    });
  };
}

export function createRenderer(canvas, options = {}) {
  if (!canvas || typeof canvas.getContext !== 'function') {
    throw new Error('createRenderer requires a canvas element');
  }
  const gl = canvas.getContext('webgl2', {alpha: false, antialias: true, preserveDrawingBuffer: true});
  if (!gl) {
    throw new Error('WebGL2 is not available in this browser');
  }
  const guard = gl.__flappyLabEntry || null; // present when created via the patched getContext
  let lastFrame = null; // {s, opts} of the last render, for corrective redraws

  if (guard) {
    guard.redraw = () => {
      if (lastFrame && !state.destroyed && !state.contextLost) {
        present(lastFrame.s, lastFrame.opts);
      }
    };
  }

  const state = {
    destroyed: false,
    contextLost: false,
    width: canvas.width || 480,
    height: canvas.height || 720,
    dpr: 1,
    highContrast: options.highContrast === true,
    reducedMotion: options.reducedMotion === true,
    palette: PALETTES.normal,
  };

  // --- GL resources (rebuilt on context restore) ---
  let program = null;
  let vertexShader = null;
  let fragmentShader = null;
  let vao = null;
  let vbo = null;
  let uColor = null;
  let vertices = [];

  function compile(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(`shader compile failed: ${gl.getShaderInfoLog(shader)}`);
    }
    return shader;
  }

  function buildResources() {
    vertexShader = compile(gl.VERTEX_SHADER, VS);
    fragmentShader = compile(gl.FRAGMENT_SHADER, FS);
    program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`program link failed: ${gl.getProgramInfoLog(program)}`);
    }
    uColor = gl.getUniformLocation(program, 'uColor');
    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  function deleteResources() {
    if (vbo) gl.deleteBuffer(vbo);
    if (vao) gl.deleteVertexArray(vao);
    if (vertexShader) gl.deleteShader(vertexShader);
    if (fragmentShader) gl.deleteShader(fragmentShader);
    if (program) gl.deleteProgram(program);
    vbo = null;
    vao = null;
    vertexShader = null;
    fragmentShader = null;
    program = null;
    uColor = null;
  }

  const onContextLost = (event) => {
    event.preventDefault(); // defers loss so we can clean up and await restore
    state.contextLost = true;
    if (guard) guard.contextLost = true;
  };
  const onContextRestored = () => {
    if (state.destroyed) return;
    buildResources();
    gl.useProgram(program);
    gl.viewport(0, 0, canvas.width, canvas.height);
    state.contextLost = false;
    if (guard) guard.contextLost = false;
  };
  canvas.addEventListener('webglcontextlost', onContextLost, false);
  canvas.addEventListener('webglcontextrestored', onContextRestored, false);

  buildResources();
  gl.useProgram(program);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.1, 0.1, 0.12, 1);

  // --- geometry helpers: logical (x, y, down-positive) -> clip space ---
  const px = (x, W) => (2 * x) / W - 1;
  const py = (y, H) => 1 - (2 * y) / H;

  function quad(x, y, w, h) {
    const x2 = x + w;
    const y2 = y + h;
    vertices.push(px(x), py(y), px(x2), py(y), px(x2), py(y2),
      px(x), py(y), px(x2), py(y2), px(x), py(y2));
  }

  function fan(cx, cy, r, segments) {
    for (let i = 0; i < segments; i += 1) {
      const a0 = (i / segments) * Math.PI * 2;
      const a1 = ((i + 1) / segments) * Math.PI * 2;
      const x0 = cx + r * Math.cos(a0);
      const y0 = cy + r * Math.sin(a0);
      const x1 = cx + r * Math.cos(a1);
      const y1 = cy + r * Math.sin(a1);
      vertices.push(px(cx), py(cy), px(x0), py(y0), px(x1), py(y1));
    }
  }

  function flush() {
    if (vertices.length === 0) return;
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.DYNAMIC_DRAW);
    gl.drawArrays(gl.TRIANGLES, 0, vertices.length / 3);
    vertices = [];
  }

  function draw(color, painter) {
    vertices.length = 0;
    painter();
    gl.uniform4f(uColor, color[0], color[1], color[2], color[3]);
    flush();
  }

  function drawScene(s, {highContrast, reducedMotion}) {
    const pal = highContrast ? PALETTES.high : PALETTES.normal;
    const cfg = (s && s.config && typeof s.config === 'object') ? s.config
      : {width: state.width, height: state.height, groundHeight: 0, birdRadius: 12, pipeWidth: 32, gapHeight: 120};
    const W = cfg.width;
    const H = cfg.height;
    const ground = cfg.groundHeight;
    const pipes = (s.pipes && s.pipes.length) ? s.pipes : [];
    const bird = (s.bird && typeof s.bird === 'object') ? s.bird : {x: W / 2, y: H / 2, vy: 0};

    draw(pal.sky, () => quad(0, 0, W, H));

    const lipH = 16;
    const lipOv = 3;
    for (const pipe of pipes) {
      const x = pipe.x;
      const w = cfg.pipeWidth;
      const gapTop = pipe.gapY;
      const gapBottom = pipe.gapY + cfg.gapHeight;
      if (gapTop > 0) draw(pal.pipe, () => quad(x, 0, w, gapTop));
      draw(pal.pipeCap, () => quad(x - lipOv, gapTop, w + lipOv * 2, lipH));
      if (gapBottom < H - ground) {
        draw(pal.pipe, () => quad(x, gapBottom, w, H - ground - gapBottom));
        draw(pal.pipeCap, () => quad(x - lipOv, gapBottom, w + lipOv * 2, lipH));
      }
    }

    draw(pal.ground, () => quad(0, H - ground, W, ground));
    draw(pal.groundEdge, () => quad(0, H - ground, W, Math.max(4, Math.round(ground * 0.06))));

    const r = cfg.birdRadius;
    let by = bird.y;
    if (s.phase === 'ready' && !reducedMotion) {
      by += Math.sin(s.frame * 0.12) * 3; // deterministic idle bob, no wall clock
    }
    draw(pal.bird, () => fan(bird.x, by, r, 20));
    draw(pal.birdShade, () => fan(bird.x - r * 0.35, by + r * 0.3, r * 0.55, 14));
    draw(pal.beak, () => {
      const bx = bird.x + r * 0.55;
      vertices.push(px(bx), py(by - r * 0.1), px(bird.x + r * 1.5), py(by - r * 0.4),
        px(bird.x + r * 1.5), py(by + r * 0.45));
      vertices.push(px(bx), py(by - r * 0.1), px(bird.x + r * 1.5), py(by + r * 0.45),
        px(bx), py(by + r * 0.55));
    });
    draw(pal.eyeWhite, () => fan(bird.x + r * 0.35, by - r * 0.4, r * 0.34, 14));
    draw(pal.pupil, () => fan(bird.x + r * 0.42, by - r * 0.4, r * 0.15, 12));

    if (s.phase === 'paused') {
      draw(pal.pauseOverlay, () => quad(0, 0, W, H));
    } else if (s.phase === 'gameover') {
      draw(pal.overOverlay, () => quad(0, 0, W, H));
    }
  }

  function resize(width, height, dpr = 1) {
    if (state.destroyed) throw new Error('renderer is destroyed');
    if (!isFiniteNum(width) || width <= 0 || !isFiniteNum(height) || height <= 0) {
      throw new RangeError('resize width/height must be positive finite numbers');
    }
    if (!isFiniteNum(dpr) || dpr <= 0 || dpr > 4) {
      throw new RangeError('dpr must be positive and <= 4');
    }
    state.width = width;
    state.height = height;
    state.dpr = dpr;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    if (!state.contextLost) {
      gl.viewport(0, 0, canvas.width, canvas.height);
    }
  }

  function present(s, opts) {
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    drawScene(s, opts);
  }

  function render(s, renderOptions = {}) {
    if (state.destroyed) throw new Error('renderer is destroyed');
    if (state.contextLost) return; // paused drawing until webglcontextrestored
    if (!s || typeof s !== 'object') {
      throw new TypeError('render requires a game state');
    }
    const opts = {
      highContrast: renderOptions.highContrast ?? state.highContrast,
      reducedMotion: renderOptions.reducedMotion ?? state.reducedMotion,
    };
    present(s, opts);
    lastFrame = {s, opts};
    if (guard) guard.fresh = true;
  }

  function info() {
    const out = {
      backend: 'webgl2',
      width: state.width,
      height: state.height,
      dpr: state.dpr,
      contextLost: state.contextLost,
      destroyed: state.destroyed,
    };
    // Optional diagnostics channel used by the in-page smoke test; present
    // only when the smoke has flagged an anomaly (never on a healthy page).
    if (typeof window !== 'undefined' && window.__flappyLabDiag &&
        window.__flappyLabDiag.enabled) {
      out.debug = window.__flappyLabDiag.summary;
    }
    return out;
  }

  function destroy() {
    if (state.destroyed) return; // idempotent
    canvas.removeEventListener('webglcontextlost', onContextLost, false);
    canvas.removeEventListener('webglcontextrestored', onContextRestored, false);
    deleteResources();
    state.destroyed = true;
    lastFrame = null;
    if (guard) {
      guard.destroyed = true;
      guard.fresh = false;
      guard.redraw = null;
    }
  }

  return {resize, render, destroy, info};
}

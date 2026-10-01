// Procedural WebGL2 renderer per specs/renderer.md.
// No Canvas2D, no network assets, no internal animation loop.

const VERT_SRC = `#version 300 es
layout(location=0) in vec2 aPos;
layout(location=1) in vec4 aColor;
uniform vec2 uSize;
out vec4 vColor;
void main() {
  vec2 clip = vec2(2.0 * aPos.x / uSize.x - 1.0, 1.0 - 2.0 * aPos.y / uSize.y);
  gl_Position = vec4(clip, 0.0, 1.0);
  vColor = aColor;
}
`;

const FRAG_SRC = `#version 300 es
precision mediump float;
in vec4 vColor;
out vec4 outColor;
void main() {
  outColor = vColor;
}
`;

const STANDARD_PALETTE = {
  sky: [0.53, 0.81, 0.92, 1],
  sun: [1, 0.84, 0.3, 1],
  cloud: [1, 1, 1, 0.9],
  ground: [0.4, 0.68, 0.35, 1],
  groundTop: [0.52, 0.8, 0.42, 1],
  pipe: [0.18, 0.6, 0.24, 1],
  pipeCap: [0.28, 0.72, 0.32, 1],
  bird: [0.98, 0.76, 0.16, 1],
  wing: [0.86, 0.55, 0.1, 1],
  eye: [1, 1, 1, 1],
  pupil: [0.1, 0.1, 0.1, 1],
  beak: [0.95, 0.5, 0.15, 1],
  overlayPaused: [0, 0, 0, 0.45],
  overlayGameOver: [0.45, 0.05, 0.05, 0.45],
};

const HIGH_CONTRAST_PALETTE = {
  sky: [0.03, 0.03, 0.03, 1],
  sun: [1, 1, 0.25, 1],
  cloud: [0.85, 0.85, 0.85, 0.9],
  ground: [0.08, 0.35, 0.08, 1],
  groundTop: [0.15, 0.5, 0.15, 1],
  pipe: [0.1, 1, 0.1, 1],
  pipeCap: [0.35, 1, 0.35, 1],
  bird: [1, 1, 0, 1],
  wing: [1, 0.6, 0, 1],
  eye: [1, 1, 1, 1],
  pupil: [0, 0, 0, 1],
  beak: [1, 0.7, 0, 1],
  overlayPaused: [0, 0, 0, 0.6],
  overlayGameOver: [0.6, 0, 0, 0.6],
};

function isFiniteNumber(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function pushRect(arr, x, y, w, h, c) {
  const p0 = [x, y];
  const p1 = [x + w, y];
  const p2 = [x + w, y + h];
  const p3 = [x, y + h];
  for (const p of [p0, p1, p2, p0, p2, p3]) {
    arr.push(p[0], p[1], c[0], c[1], c[2], c[3]);
  }
}

function pushCircle(arr, cx, cy, r, c, segments = 24) {
  let ax = cx + r;
  let ay = cy;
  for (let i = 1; i <= segments; i++) {
    const t = (i * Math.PI * 2) / segments;
    const bx = cx + r * Math.cos(t);
    const by = cy + r * Math.sin(t);
    arr.push(cx, cy, c[0], c[1], c[2], c[3]);
    arr.push(ax, ay, c[0], c[1], c[2], c[3]);
    arr.push(bx, by, c[0], c[1], c[2], c[3]);
    ax = bx;
    ay = by;
  }
}

function buildScene(arr, state, pal) {
  const cfg = state.config;
  const w = cfg.width;
  const h = cfg.height;
  const groundY = h - cfg.groundHeight;
  pushRect(arr, 0, 0, w, h, pal.sky);
  pushCircle(arr, w - 90, 80, 42, pal.sun);
  pushCircle(arr, w * 0.25, 90, 26, pal.cloud);
  pushCircle(arr, w * 0.25 + 26, 100, 22, pal.cloud);
  pushCircle(arr, w * 0.55, 140, 20, pal.cloud);
  if (cfg.groundHeight > 0) {
    pushRect(arr, 0, groundY, w, cfg.groundHeight, pal.ground);
    pushRect(arr, 0, groundY, w, Math.min(10, cfg.groundHeight), pal.groundTop);
  }
  const pipes = Array.isArray(state.pipes) ? state.pipes : [];
  for (const pipe of pipes) {
    const gapBottom = pipe.gapY + cfg.gapHeight;
    pushRect(arr, pipe.x, 0, cfg.pipeWidth, pipe.gapY, pal.pipe);
    if (gapBottom < groundY) {
      pushRect(arr, pipe.x, gapBottom, cfg.pipeWidth, groundY - gapBottom, pal.pipe);
    }
    pushRect(arr, pipe.x - 5, pipe.gapY - 18, cfg.pipeWidth + 10, 18, pal.pipeCap);
    pushRect(arr, pipe.x - 5, gapBottom, cfg.pipeWidth + 10, 18, pal.pipeCap);
  }
  const bird = state.bird;
  const r = cfg.birdRadius;
  pushCircle(arr, bird.x, bird.y, r * 1.5, pal.bird);
  pushCircle(arr, bird.x - r * 0.5, bird.y + r * 0.25, r * 0.8, pal.wing);
  pushCircle(arr, bird.x + r * 0.45, bird.y - r * 0.35, r * 0.35, pal.eye);
  pushCircle(arr, bird.x + r * 0.52, bird.y - r * 0.35, r * 0.16, pal.pupil);
  pushRect(arr, bird.x + r * 0.9, bird.y - r * 0.2, r * 0.8, r * 0.5, pal.beak);
  if (state.phase === 'paused') {
    pushRect(arr, 0, 0, w, h, pal.overlayPaused);
  } else if (state.phase === 'gameover') {
    pushRect(arr, 0, 0, w, h, pal.overlayGameOver);
  }
}

export function createRenderer(canvas, options = {}) {
  if (canvas === null || typeof canvas !== 'object' || typeof canvas.getContext !== 'function') {
    throw new TypeError('createRenderer requires a canvas element');
  }
  const gl = canvas.getContext('webgl2');
  if (!gl) {
    throw new Error('WebGL2 is not available for this canvas');
  }
  const opts = options === null ? {} : options;
  const highContrast = opts.highContrast === true;
  const pal = highContrast ? HIGH_CONTRAST_PALETTE : STANDARD_PALETTE;

  let logicalWidth = canvas.clientWidth || canvas.width || 300;
  let logicalHeight = canvas.clientHeight || canvas.height || 150;
  let dpr = 1;
  let contextLost = false;
  let destroyed = false;
  let program = null;
  let vbo = null;
  let uSize = null;

  function compileShader(type, src) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error('shader compilation failed: ' + log);
    }
    return shader;
  }

  function buildResources() {
    const vs = compileShader(gl.VERTEX_SHADER, VERT_SRC);
    const fs = compileShader(gl.FRAGMENT_SHADER, FRAG_SRC);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      gl.deleteProgram(prog);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      throw new Error('program link failed: ' + log);
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    program = prog;
    uSize = gl.getUniformLocation(prog, 'uSize');
    vbo = gl.createBuffer();
  }

  buildResources();

  const onContextLost = (event) => {
    event.preventDefault();
    contextLost = true;
  };
  const onContextRestored = () => {
    if (destroyed) {
      return;
    }
    buildResources();
    contextLost = false;
  };
  canvas.addEventListener('webglcontextlost', onContextLost, false);
  canvas.addEventListener('webglcontextrestored', onContextRestored, false);

  return {
    resize(width, height, nextDpr = 1) {
      if (destroyed) {
        throw new Error('renderer has been destroyed');
      }
      if (!isFiniteNumber(width) || width <= 0 || !isFiniteNumber(height) || height <= 0 ||
          !isFiniteNumber(nextDpr) || nextDpr <= 0 || nextDpr > 4) {
        throw new RangeError('resize requires finite positive width/height and a finite dpr at most 4');
      }
      canvas.width = Math.round(width * nextDpr);
      canvas.height = Math.round(height * nextDpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      gl.viewport(0, 0, canvas.width, canvas.height);
      logicalWidth = width;
      logicalHeight = height;
      dpr = nextDpr;
    },
    render(state, renderOptions = {}) {
      if (destroyed) {
        throw new Error('renderer has been destroyed');
      }
      if (contextLost) {
        return;
      }
      if (state === null || typeof state !== 'object' ||
          state.config === null || typeof state.config !== 'object') {
        throw new TypeError('render requires a game state with a config object');
      }
      const cfg = state.config;
      const vertices = [];
      buildScene(vertices, state, pal);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(pal.sky[0], pal.sky[1], pal.sky[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.useProgram(program);
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.DYNAMIC_DRAW);
      const stride = 24;
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 4, gl.FLOAT, false, stride, 8);
      gl.uniform2f(uSize, cfg.width, cfg.height);
      gl.drawArrays(gl.TRIANGLES, 0, vertices.length / 6);
    },
    info() {
      return {
        backend: 'webgl2',
        width: logicalWidth,
        height: logicalHeight,
        dpr,
        contextLost,
        destroyed,
      };
    },
    destroy() {
      if (destroyed) {
        return;
      }
      destroyed = true;
      canvas.removeEventListener('webglcontextlost', onContextLost, false);
      canvas.removeEventListener('webglcontextrestored', onContextRestored, false);
      if (program) {
        gl.deleteProgram(program);
        program = null;
      }
      if (vbo) {
        gl.deleteBuffer(vbo);
        vbo = null;
      }
    },
  };
}

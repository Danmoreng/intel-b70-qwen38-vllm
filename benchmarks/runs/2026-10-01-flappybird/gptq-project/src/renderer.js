// Public API contract: specs/renderer.md
// Procedural WebGL2 renderer; logical pixels have a top-left origin and every
// vertex is converted exactly once to clip space:
//   clipX = 2*x/logicalWidth - 1, clipY = 1 - 2*y/logicalHeight

const VS = `#version 300 es
layout(location=0) in vec2 aPos;
layout(location=1) in vec3 aColor;
out vec3 vColor;
void main() {
  vColor = aColor;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision mediump float;
in vec3 vColor;
out vec4 outColor;
void main() {
  outColor = vec4(vColor, 1.0);
}`;

// palette: [sky, ground, grass, pipe, pipeCap, bird, belly, beak, eye, cloud]
const PALETTES = {
  normal: [
    [0.45, 0.73, 0.92],
    [0.78, 0.61, 0.35],
    [0.29, 0.64, 0.29],
    [0.16, 0.66, 0.3],
    [0.11, 0.52, 0.22],
    [1.0, 0.79, 0.16],
    [0.98, 0.94, 0.8],
    [0.98, 0.56, 0.1],
    [0.1, 0.1, 0.1],
    [0.97, 0.97, 1.0]
  ],
  highContrast: [
    [0.04, 0.06, 0.14],
    [0.16, 0.16, 0.16],
    [0.55, 1.0, 0.45],
    [1.0, 0.12, 0.12],
    [0.55, 0.03, 0.03],
    [1.0, 1.0, 1.0],
    [0.6, 0.6, 0.6],
    [1.0, 0.35, 0.0],
    [0.0, 0.0, 0.0],
    [0.85, 0.85, 0.85]
  ]
};

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`shader compilation failed: ${log}`);
  }
  return shader;
}

export function createRenderer(canvas, options = {}) {
  if (canvas === null || canvas === undefined || typeof canvas.getContext !== 'function') {
    throw new Error('a canvas element is required');
  }
  const gl = canvas.getContext('webgl2');
  if (!gl) {
    throw new Error('WebGL2 is not available; this renderer requires an actual WebGL2 context');
  }
  const baseHighContrast = options.highContrast === true;

  let program = null;
  let vs = null;
  let fs = null;
  let vbo = null;
  let width = 0;
  let height = 0;
  let dpr = 1;
  let contextLost = false;
  let destroyed = false;
  let hasSize = false;

  function initGL() {
    if (vs === null) {
      vs = compile(gl, gl.VERTEX_SHADER, VS);
      fs = compile(gl, gl.FRAGMENT_SHADER, FS);
    }
    if (program === null) {
      program = gl.createProgram();
      gl.attachShader(program, vs);
      gl.attachShader(program, fs);
      gl.bindAttribLocation(program, 0, 'aPos');
      gl.bindAttribLocation(program, 1, 'aColor');
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(`program link failed: ${gl.getProgramInfoLog(program)}`);
      }
    }
    if (vbo === null) {
      vbo = gl.createBuffer();
    }
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    const stride = 5 * Float32Array.BYTES_PER_ELEMENT;
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 2 * Float32Array.BYTES_PER_ELEMENT);
  }

  function onLost(e) {
    if (e && typeof e.preventDefault === 'function') {
      e.preventDefault();
    }
    contextLost = true;
  }
  function onRestored() {
    if (destroyed) {
      return;
    }
    try {
      initGL();
      contextLost = false;
    } catch {
      contextLost = true;
    }
  }
  canvas.addEventListener('webglcontextlost', onLost, false);
  canvas.addEventListener('webglcontextrestored', onRestored, false);
  initGL();

  // Logical (top-left origin) to clip space; called exactly once per vertex.
  function clip(x, y, W, H) {
    return [2 * x / W - 1, 1 - 2 * y / H];
  }

  function resize(w, h, dprValue = 1) {
    if (destroyed) {
      throw new Error('renderer is destroyed');
    }
    for (const [name, v] of [['width', w], ['height', h], ['dpr', dprValue]]) {
      if (typeof v !== 'number' || !Number.isFinite(v) || v <= 0) {
        throw new RangeError(`renderer ${name} must be a finite positive number`);
      }
    }
    if (dprValue > 4) {
      throw new RangeError('renderer dpr must be <= 4');
    }
    width = w;
    height = h;
    dpr = dprValue;
    hasSize = true;
    canvas.width = Math.round(w * dprValue);
    canvas.height = Math.round(h * dprValue);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    gl.viewport(0, 0, canvas.width, canvas.height);
  }

  // Scene building helpers over a flat [x, y, r, g, b] ... interleaved vertex array.
  function pushRect(verts, x0, y0, x1, y1, color, W, H) {
    const a = clip(x0, y0, W, H);
    const b = clip(x1, y0, W, H);
    const c = clip(x1, y1, W, H);
    const d = clip(x0, y1, W, H);
    for (const p of [a, d, b, b, d, c]) {
      verts.push(p[0], p[1], color[0], color[1], color[2]);
    }
  }

  // Convex polygon (circle approximation) as a triangle fan from its center.
  function pushPoly(verts, cx, cy, rx, ry, n, color, W, H, dx = 0, dy = 0) {
    const center = clip(cx + dx, cy + dy, W, H);
    const ring = [];
    for (let i = 0; i < n; i++) {
      const t = (i / n) * Math.PI * 2;
      ring.push(clip(cx + dx + Math.cos(t) * rx, cy + dy + Math.sin(t) * ry, W, H));
    }
    for (let i = 0; i < n; i++) {
      verts.push(center[0], center[1], color[0], color[1], color[2]);
      for (const p of [ring[i], ring[(i + 1) % n]]) {
        verts.push(p[0], p[1], color[0], color[1], color[2]);
      }
    }
  }

  function pushTriangle(verts, p1, p2, p3, color, W, H) {
    for (const p of [p1, p2, p3]) {
      const c = clip(p[0], p[1], W, H);
      verts.push(c[0], c[1], color[0], color[1], color[2]);
    }
  }

  function draw(state, highContrast) {
    if (destroyed || contextLost) {
      return;
    }
    if (!hasSize) {
      resize(state.config.width, state.config.height, 1);
    }
    const W = state.config.width;
    const H = state.config.height;
    const pal = highContrast ? PALETTES.highContrast : PALETTES.normal;
    const [sky, ground, grass, pipe, pipeCap, bird, belly, beak, eye, cloud] = pal;
    const groundY = H - state.config.groundHeight;
    const verts = [];

    pushRect(verts, 0, 0, W, H, sky, W, H);
    // Decorative clouds keep the sky region distinct in every phase.
    pushPoly(verts, 0.25 * W, 0.18 * H, 26, 14, 12, cloud, W, H);
    pushPoly(verts, 0.62 * W, 0.12 * H, 32, 16, 12, cloud, W, H);
    pushPoly(verts, 0.85 * W, 0.3 * H, 22, 12, 12, cloud, W, H);
    pushRect(verts, 0, groundY, W, H, ground, W, H);
    pushRect(verts, 0, groundY, W, groundY + 10, grass, W, H);

    const pw = state.config.pipeWidth;
    const gapH = state.config.gapHeight;
    const capH = 26;
    for (const p of state.pipes) {
      const left = p.x;
      const right = left + pw;
      pushRect(verts, left, 0, right, p.gapY, pipe, W, H);
      pushRect(verts, left - 4, p.gapY - capH, right + 4, p.gapY, pipeCap, W, H);
      pushRect(verts, left, p.gapY + gapH, right, groundY, pipe, W, H);
      pushRect(verts, left - 4, p.gapY + gapH, right + 4, p.gapY + gapH + capH, pipeCap, W, H);
    }

    // Bird at the simulated position in every phase (ready included).
    const {x, y} = state.bird;
    const r = state.config.birdRadius;
    pushPoly(verts, x, y, r * 1.15, r, 18, bird, W, H);
    pushPoly(verts, x, y + 0.2 * r, r * 0.7, r * 0.55, 12, belly, W, H, 0, 0.25 * r);
    pushTriangle(verts, [x + 0.5 * r, y], [x + 1.35 * r, y - 0.28 * r], [x + 1.35 * r, y + 0.28 * r], beak, W, H);
    pushPoly(verts, x, y, r * 0.2, r * 0.2, 8, eye, W, H, 0.55 * r, -0.45 * r);
    if (state.phase === 'gameover') {
      // X eyes: two small dark squares over the eye region read as "down".
      pushRect(verts, x + 0.3 * r, y - 0.65 * r, x + 0.8 * r, y - 0.25 * r, eye, W, H);
    }

    gl.clearColor(sky[0], sky[1], sky[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (verts.length === 0) {
      return;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.DYNAMIC_DRAW);
    // Vertices are interleaved as 2 position + 3 color floats; count = length/5.
    gl.drawArrays(gl.TRIANGLES, 0, verts.length / 5);
  }

  function info() {
    return {
      backend: 'webgl2',
      width,
      height,
      dpr,
      contextLost,
      destroyed
    };
  }

  function destroy() {
    if (destroyed) {
      return;
    }
    destroyed = true;
    canvas.removeEventListener('webglcontextlost', onLost, false);
    canvas.removeEventListener('webglcontextrestored', onRestored, false);
    if (vbo !== null) {
      gl.deleteBuffer(vbo);
      vbo = null;
    }
    if (program !== null) {
      gl.deleteProgram(program);
      program = null;
    }
    if (vs !== null) {
      gl.deleteShader(vs);
      vs = null;
    }
    if (fs !== null) {
      gl.deleteShader(fs);
      fs = null;
    }
  }

  return {
    resize,
    render(state, {highContrast = false, reducedMotion = false} = {}) {
      if (destroyed) {
        throw new Error('renderer is destroyed');
      }
      draw(state, highContrast || baseHighContrast);
    },
    destroy,
    info
  };
}

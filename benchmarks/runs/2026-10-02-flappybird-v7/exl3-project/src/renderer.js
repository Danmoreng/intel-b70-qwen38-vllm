// Public API contract: specs/renderer.md

const VERT_SRC = `#version 300 es
layout(location=0) in vec2 a_pos;
layout(location=1) in vec4 a_color;
out vec4 v_color;
uniform vec2 u_dim;
void main() {
  // a_pos is in logical pixel coordinates (top-left origin). Converted
  // exactly once to clip coordinates:
  vec2 c = vec2(2.0 * a_pos.x / u_dim.x - 1.0, 1.0 - 2.0 * a_pos.y / u_dim.y);
  gl_Position = vec4(c, 0.0, 1.0);
  v_color = a_color;
}
`;

const FRAG_SRC = `#version 300 es
precision highp float;
in vec4 v_color;
out vec4 frag_color;
void main() {
  frag_color = v_color;
}
`;

const NORMAL_PALETTE = {
  sky: [0.53, 0.81, 0.92, 1],
  cloud: [0.96, 0.97, 0.98, 1],
  ground: [0.42, 0.65, 0.25, 1],
  groundTop: [0.55, 0.78, 0.35, 1],
  pipe: [0.36, 0.72, 0.26, 1],
  pipeCap: [0.24, 0.55, 0.18, 1],
  bird: [0.98, 0.81, 0.16, 1],
  beak: [0.94, 0.45, 0.15, 1],
  eyeWhite: [1, 1, 1, 1],
  pupil: [0.1, 0.1, 0.1, 1],
  overlay: [0, 0, 0, 1]
};

const HIGH_CONTRAST_PALETTE = {
  sky: [1, 1, 0.7, 1],
  cloud: [0.6, 0.6, 0.6, 1],
  ground: [0, 0, 0, 1],
  groundTop: [0.2, 0.2, 0.2, 1],
  pipe: [0.85, 0, 0, 1],
  pipeCap: [0.5, 0, 0, 1],
  bird: [1, 1, 1, 1],
  beak: [0.8, 0.3, 0, 1],
  eyeWhite: [0, 0, 0, 1],
  pupil: [1, 1, 1, 1],
  overlay: [0, 0, 0, 1]
};

const FLOATS_PER_VERTEX = 6; // x, y, r, g, b, a

function isFinitePositive(value) {
  return typeof value === 'number' && Number.isFinite(value) && value > 0;
}

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error('WebGL shader compilation failed: ' + log);
  }
  return shader;
}

function makeProgram(gl) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, VERT_SRC);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error('WebGL program link failed: ' + log);
  }
  return program;
}

function pushColor(verts, x, y, color) {
  verts.push(x, y, color[0], color[1], color[2], color[3]);
}

function pushQuad(verts, x0, y0, x1, y1, color) {
  pushColor(verts, x0, y0, color);
  pushColor(verts, x1, y0, color);
  pushColor(verts, x1, y1, color);
  pushColor(verts, x0, y0, color);
  pushColor(verts, x1, y1, color);
  pushColor(verts, x0, y1, color);
}

function pushEllipse(verts, cx, cy, rx, ry, segments, color) {
  for (let i = 0; i < segments; i++) {
    const a0 = (i / segments) * Math.PI * 2;
    const a1 = ((i + 1) / segments) * Math.PI * 2;
    pushColor(verts, cx, cy, color);
    pushColor(verts, cx + Math.cos(a0) * rx, cy + Math.sin(a0) * ry, color);
    pushColor(verts, cx + Math.cos(a1) * rx, cy + Math.sin(a1) * ry, color);
  }
}

export function createRenderer(canvas, options = {}) {
  if (canvas === null || typeof canvas !== 'object' || typeof canvas.getContext !== 'function') {
    throw new Error('createRenderer expects a WebGL2 canvas element');
  }
  const baseOptions = options === null ? {} : options;
  if (typeof baseOptions !== 'object' || Array.isArray(baseOptions)) {
    throw new Error('renderer options must be an object');
  }
  const gl = canvas.getContext('webgl2');
  if (!gl) {
    throw new Error('WebGL2 is not available for this canvas');
  }

  let width = 0;
  let height = 0;
  let dpr = 1;
  let destroyed = false;
  let contextLost = false;
  let program = null;
  let vao = null;
  let buffer = null;
  let uDim = null;

  function buildResources() {
    program = makeProgram(gl);
    uDim = gl.getUniformLocation(program, 'u_dim');
    vao = gl.createVertexArray();
    buffer = gl.createBuffer();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    // Interleaved layout: [x, y, r, g, b, a]; full 24-byte stride.
    const stride = FLOATS_PER_VERTEX * 4;
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, stride, 8);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  function deleteResources() {
    if (vao !== null) {
      gl.deleteVertexArray(vao);
      vao = null;
    }
    if (buffer !== null) {
      gl.deleteBuffer(buffer);
      buffer = null;
    }
    if (program !== null) {
      gl.deleteProgram(program);
      program = null;
    }
    uDim = null;
  }

  function onContextLost(event) {
    if (event && typeof event.preventDefault === 'function') {
      event.preventDefault();
    }
    contextLost = true;
  }

  function onContextRestored() {
    contextLost = false;
    buildResources();
    gl.viewport(0, 0, canvas.width, canvas.height);
  }

  canvas.addEventListener('webglcontextlost', onContextLost, false);
  canvas.addEventListener('webglcontextrestored', onContextRestored, false);

  function resize(newWidth, newHeight, newDpr = 1) {
    if (destroyed) {
      throw new Error('renderer is destroyed');
    }
    if (!isFinitePositive(newWidth) || !isFinitePositive(newHeight)) {
      throw new Error('resize width/height must be finite positive numbers');
    }
    if (!isFinitePositive(newDpr) || newDpr > 4) {
      throw new Error('resize dpr must be a finite positive number <= 4');
    }
    width = newWidth;
    height = newHeight;
    dpr = newDpr;
    canvas.width = Math.round(newWidth * newDpr);
    canvas.height = Math.round(newHeight * newDpr);
    if (canvas.style) {
      canvas.style.width = newWidth + 'px';
      canvas.style.height = newHeight + 'px';
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
  }

  function render(state, renderOptions = {}) {
    if (destroyed) {
      throw new Error('renderer is destroyed');
    }
    if (contextLost || state === null || typeof state !== 'object') {
      return;
    }
    const opts = renderOptions === null ? {} : renderOptions;
    const highContrast = Boolean(
      (opts && typeof opts === 'object' ? opts.highContrast : undefined) || baseOptions.highContrast
    );
    const palette = highContrast ? HIGH_CONTRAST_PALETTE : NORMAL_PALETTE;
    const cfg = state.config;
    if (cfg === null || typeof cfg !== 'object') {
      return;
    }
    const W = cfg.width;
    const H = cfg.height;
    const groundH = cfg.groundHeight;

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(palette.sky[0], palette.sky[1], palette.sky[2], palette.sky[3]);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const verts = [];
    // Sky (also the clear color).
    pushQuad(verts, 0, 0, W, H, palette.sky);
    // Procedural clouds.
    const reducedMotion = Boolean(opts && typeof opts === 'object' ? opts.reducedMotion : false);
    pushEllipse(verts, W * 0.25, H * 0.18, Math.max(24, W * 0.09), Math.max(12, H * 0.025), 20, palette.cloud);
    pushEllipse(verts, W * 0.7, H * 0.12, Math.max(20, W * 0.07), Math.max(10, H * 0.02), 20, palette.cloud);
    if (!reducedMotion) {
      pushEllipse(verts, W * 0.5, H * 0.28, Math.max(16, W * 0.05), Math.max(8, H * 0.015), 16, palette.cloud);
    }
    // Pipes behind the bird.
    for (const pipe of state.pipes || []) {
      const pw = cfg.pipeWidth;
      const gapH = cfg.gapHeight;
      pushQuad(verts, pipe.x, 0, pipe.x + pw, pipe.gapY, palette.pipe);
      pushQuad(verts, pipe.x - 3, pipe.gapY - 14, pipe.x + pw + 3, pipe.gapY, palette.pipeCap);
      pushQuad(verts, pipe.x, pipe.gapY + gapH, pipe.x + pw, H - groundH, palette.pipe);
      pushQuad(verts, pipe.x - 3, pipe.gapY + gapH, pipe.x + pw + 3, pipe.gapY + gapH + 14, palette.pipeCap);
    }
    // Ground.
    pushQuad(verts, 0, H - groundH, W, H, palette.ground);
    pushQuad(verts, 0, H - groundH, W, H - groundH + 8, palette.groundTop);
    // Bird.
    const bird = state.bird;
    if (bird !== null && typeof bird === 'object') {
      const r = cfg.birdRadius;
      pushEllipse(verts, bird.x, bird.y, r, r, 24, palette.bird);
      // Beak: triangle to the right of the body.
      pushColor(verts, bird.x + r * 0.4, bird.y - r * 0.35, palette.beak);
      pushColor(verts, bird.x + r * 0.4, bird.y + r * 0.35, palette.beak);
      pushColor(verts, bird.x + r * 1.6, bird.y, palette.beak);
      // Eye.
      pushEllipse(verts, bird.x + r * 0.35, bird.y - r * 0.35, r * 0.45, r * 0.45, 16, palette.eyeWhite);
      pushEllipse(verts, bird.x + r * 0.45, bird.y - r * 0.35, r * 0.2, r * 0.2, 12, palette.pupil);
    }
    // Subtle overlay to distinguish paused/gameover scenes.
    if (state.phase === 'paused' || state.phase === 'gameover') {
      // Draw with translucent-looking tint: use a lighter dark tone via a
      // second sky-dark quad is not possible without blending; darken sky band only.
      pushQuad(verts, 0, 0, W, 40, palette.overlay);
    }

    gl.useProgram(program);
    gl.uniform2f(uDim, W, H);
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.DYNAMIC_DRAW);
    gl.drawArrays(gl.TRIANGLES, 0, verts.length / FLOATS_PER_VERTEX);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
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
    canvas.removeEventListener('webglcontextlost', onContextLost, false);
    canvas.removeEventListener('webglcontextrestored', onContextRestored, false);
    if (!contextLost) {
      deleteResources();
    } else {
      vao = null;
      buffer = null;
      program = null;
      uDim = null;
    }
    destroyed = true;
  }

  buildResources();

  return {resize, render, destroy, info};
}
